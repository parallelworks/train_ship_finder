import sys, os, json, time
from random import randint
import argparse

import parsl
print(parsl.__version__, flush = True)
from parsl.app.app import python_app, bash_app
from parsl.config import Config
from parsl.channels import SSHChannel
from parsl.providers import LocalProvider
from parsl.executors import HighThroughputExecutor

import parsl_utils

def read_args():
    parser=argparse.ArgumentParser()
    parsed, unknown = parser.parse_known_args()
    for arg in unknown:
        if arg.startswith(("-", "--")):
            parser.add_argument(arg)
    pwargs=vars(parser.parse_args())
    print(pwargs)
    return pwargs

with open('executors.json', 'r') as f:
    exec_conf = json.load(f)

@parsl_utils.parsl_wrappers.log_app
@parsl_utils.parsl_wrappers.stage_app(exec_conf['cpu_executor']['HOST_IP'])
@bash_app(executors=['cpu_executor'])
def generate_data(run_dir, path_to_sing, gen_script, imgdir, num_samples, max_noise, max_brightness_shift, rotation_range,
        horizontal_flip, vertical_flip, zca_whitening, inputs_dict = {}, outputs_dict = {}, stdout='std.out', stderr = 'std.err'):
    return '''
        cd {run_dir}
        singularity exec -B `pwd`:`pwd` -B {run_dir}:{run_dir} {path_to_sing} /usr/local/bin/python {gen_script} \
            --imgdir {imgdir} \
            --num_samples {num_samples} \
            --max_noise {max_noise} \
            --max_brightness_shift {max_brightness_shift} \
            --rotation_range {rotation_range} \
            --horizontal_flip {horizontal_flip} \
            --vertical_flip {vertical_flip} \
            --zca_whitening {zca_whitening}
    '''.format(
        run_dir = run_dir,
        path_to_sing = path_to_sing,
        gen_script = gen_script,
        imgdir = imgdir,
        num_samples = num_samples,
        max_noise = max_noise,
        max_brightness_shift = max_brightness_shift,
        rotation_range = rotation_range,
        horizontal_flip = horizontal_flip,
        vertical_flip = vertical_flip,
        zca_whitening = zca_whitening
    )


@parsl_utils.parsl_wrappers.log_app
@parsl_utils.parsl_wrappers.stage_app(exec_conf['gpu_executor']['HOST_IP'])
@bash_app(executors=['gpu_executor'])
def train_model(run_dir, path_to_sing, train_script, imgdir, epochs, patience, batch_size, learning_rate,
                momentum, model_dir, inputs_dict = {}, outputs_dict = {}, stdout='std.out', stderr = 'std.err'):
    return '''
        nvidia-smi
        cd {run_dir}
        singularity exec --nv -B `pwd`:`pwd` -B {run_dir}:{run_dir} {path_to_sing} /usr/local/bin/python {train_script} \
            --imgdir {imgdir} \
            --epochs {epochs} \
            --batch_size {batch_size} \
            --patience {patience} \
            --learning_rate {learning_rate} \
            --momentum {momentum} \
            --model_dir {model_dir} \
            --min_ship_score 0.5 \
    '''.format(
        run_dir = run_dir,
        path_to_sing = path_to_sing,
        train_script = train_script,
        imgdir = imgdir,
        epochs = epochs,
        patience = patience,
        batch_size = batch_size,
        learning_rate = learning_rate,
        momentum = momentum,
        model_dir = model_dir
    )



if __name__ == '__main__':
    args = read_args()

    # Add sandbox directory
    for exec_label, exec_conf_i in exec_conf.items():
        if 'RUN_DIR' in exec_conf_i:
            exec_conf[exec_label]['RUN_DIR'] = os.path.join(exec_conf_i['RUN_DIR'], 'run-' + str(randint(0, 99999)).zfill(5))
        else:
            base_dir = '/contrib/{PW_USER}/tmp'.format(PW_USER = os.environ['PW_USER'])
            exec_conf[exec_label]['RUN_DIR'] = os.path.join(base_dir, 'run-' + str(randint(0, 99999)).zfill(5))

    config = Config(
        executors = [
            HighThroughputExecutor(
                worker_ports = ((int(exec_conf['gpu_executor']['WORKER_PORT_1']), int(exec_conf['gpu_executor']['WORKER_PORT_2']))),
                label = 'gpu_executor',
                worker_debug = True,             # Default False for shorter logs
                cores_per_worker = int(exec_conf['gpu_executor']['CORES_PER_WORKER']), # One worker per node
                worker_logdir_root = exec_conf['gpu_executor']['WORKER_LOGDIR_ROOT'],  #os.getcwd() + '/parsllogs',
                provider = LocalProvider(
                    worker_init = 'source {conda_sh}; conda activate {conda_env}; cd {run_dir}'.format(
                        conda_sh = os.path.join(exec_conf['gpu_executor']['REMOTE_CONDA_DIR'], 'etc/profile.d/conda.sh'),
                        conda_env = exec_conf['gpu_executor']['REMOTE_CONDA_ENV'],
                        run_dir = exec_conf['gpu_executor']['RUN_DIR']
                    ),
                    channel = SSHChannel(
                        hostname = exec_conf['gpu_executor']['HOST_IP'],
                        username = os.environ['PW_USER'],
                        script_dir = exec_conf['gpu_executor']['SSH_CHANNEL_SCRIPT_DIR'], # Full path to a script dir where generated scripts could be sent to
                        key_filename = '/home/{PW_USER}/.ssh/pw_id_rsa'.format(PW_USER = os.environ['PW_USER'])
                    )
                )
            ),
            HighThroughputExecutor(
                worker_ports = ((int(exec_conf['cpu_executor']['WORKER_PORT_1']), int(exec_conf['cpu_executor']['WORKER_PORT_2']))),
                label = 'cpu_executor',
                worker_debug = True,             # Default False for shorter logs
                cores_per_worker = int(exec_conf['cpu_executor']['CORES_PER_WORKER']), # One worker per node
                worker_logdir_root = exec_conf['cpu_executor']['WORKER_LOGDIR_ROOT'],  #os.getcwd() + '/parsllogs',
                provider = LocalProvider(
                    worker_init = 'source {conda_sh}; conda activate {conda_env}; cd {run_dir}'.format(
                        conda_sh = os.path.join(exec_conf['cpu_executor']['REMOTE_CONDA_DIR'], 'etc/profile.d/conda.sh'),
                        conda_env = exec_conf['cpu_executor']['REMOTE_CONDA_ENV'],
                        run_dir = exec_conf['cpu_executor']['RUN_DIR']
                    ),
                    channel = SSHChannel(
                        hostname = exec_conf['cpu_executor']['HOST_IP'],
                        username = os.environ['PW_USER'],
                        script_dir = exec_conf['cpu_executor']['SSH_CHANNEL_SCRIPT_DIR'], # Full path to a script dir where generated scripts could be sent to
                        key_filename = '/home/{PW_USER}/.ssh/pw_id_rsa'.format(PW_USER = os.environ['PW_USER'])
                    )
                )
            )
        ]
    )

    print('Loading Parsl Config', flush = True)
    parsl.load(config)

    # Generate data:
    gen_data_fut = generate_data(
        exec_conf['cpu_executor']['RUN_DIR'],
        exec_conf['cpu_executor']['SINGULARITY_CONTAINER_PATH'],
        './generate_data.py',
        os.path.basename(args['imgdir']),
        args['num_extra_samples'],
        args['max_noise'], args['max_brightness_shift'], args['rotation_range'],
        args['horizontal_flip'], args['vertical_flip'], args['zca_whitening'],
        inputs_dict = {
            "gen_script": {
                "type": "file",
                "global_path": "pw://{cwd}/train_ship_finder/generate_data.py",
                "worker_path": "{remote_dir}/generate_data.py".format(remote_dir =  exec_conf['cpu_executor']['RUN_DIR'])
            },
            "imgdir": {
                "type": "directory",
                "global_path": args['imgdir'],
                "worker_path": "{remote_dir}/{data_dir}".format(
                    remote_dir = exec_conf['cpu_executor']['RUN_DIR'],
                    data_dir = os.path.basename(args['imgdir'])
                )
            },
        },
        outputs_dict = {
            "imgdir_gen": {
                "type": "directory",
                "global_path": args['imgdir_gen'],
                "worker_path": "{remote_dir}/{data_dir}".format(
                    remote_dir = exec_conf['cpu_executor']['RUN_DIR'],
                    data_dir = os.path.basename(args['imgdir'])
                )
            }
        },
        stdout = os.path.join(exec_conf['cpu_executor']['RUN_DIR'], 'std.out'),
        stderr = os.path.join(exec_conf['cpu_executor']['RUN_DIR'], 'std.err')
    )

    gen_data_fut.result()

    # Train model:
    train_model_fut = train_model(
        exec_conf['gpu_executor']['RUN_DIR'],
        exec_conf['gpu_executor']['SINGULARITY_CONTAINER_PATH'],
        './train.py',
        os.path.basename(args['imgdir_gen']),
        args['epochs'], args['patience'], args['batch_size'], args['learning_rate'], args['momentum'],
        os.path.basename(args['model_dir']),
        inputs_dict = {
            "train_script": {
                "type": "file",
                "global_path": "pw://{cwd}/train_ship_finder/train.py",
                "worker_path": "{remote_dir}/train.py".format(remote_dir =  exec_conf['gpu_executor']['RUN_DIR'])
            },
            "imgdir_gen": {
                "type": "directory",
                "global_path": args['imgdir_gen'],
                "worker_path": "{remote_dir}/{data_dir}".format(
                    remote_dir = exec_conf['gpu_executor']['RUN_DIR'],
                    data_dir = os.path.basename(args['imgdir_gen'])
                )
            },
        },
        outputs_dict = {
            "model_dir": {
                "type": "directory",
                "global_path": args['model_dir'],
                "worker_path": "{remote_dir}/{model_dir}".format(
                    remote_dir = exec_conf['gpu_executor']['RUN_DIR'],
                    model_dir = os.path.basename(args['model_dir'])
                )
            }
        },
        stdout = os.path.join(exec_conf['gpu_executor']['RUN_DIR'], 'std.out'),
        stderr = os.path.join(exec_conf['gpu_executor']['RUN_DIR'], 'std.err')
    )

    train_model_fut.result()
