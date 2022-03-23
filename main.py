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

with open('executors.json', 'r') as f:
    exec_conf = json.load(f)

@parsl_utils.parsl_wrappers.log_app
@parsl_utils.parsl_wrappers.stage_app(exec_conf['cpu_executor']['HOST_IP'])
@bash_app(executors=['cpu_executor'])
def generate_data(run_dir, gen_script, imgdir, num_samples, max_noise, max_brightness_shift, rotation_range,
        horizontal_flip, vertical_flip, zca_whitening, inputs_dict = {}, outputs_dict = {}, stdout='std.out', stderr = 'std.err'):
    return '''
        singularity exec --nv -B `pwd`:`pwd` -B {run_dir}:{run_dir} {path_to_sing} /usr/local/bin/python {gen_script} \
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
        path_to_sing = exec_conf['cpu_executor']['SINGULARITY_CONTAINER_PATH'],
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
def train_model(run_dir, pyscript, imgdir, epochs, batch_size, learning_rate,
                momentum, model_dir, inputs_dict = {}, outputs_dict = {}, stdout='std.out', stderr = 'std.err'):
    return '''
        singularity exec --nv -B `pwd`:`pwd` -B {run_dir}:{run_dir} {path_to_sing} /usr/local/bin/python {train_script} \
            --imgdir {imgdir} \
            --epochs {epochs} \
            --batch_size {batch_size} \
            --learning_rate {learning_rate} \
            --momentum {momentum} \
            --model_dir {model_dir}
    '''.format(
        run_dir = run_dir,
        path_to_sing = exec_conf['gpu_executor']['SINGULARITY_CONTAINER_PATH'],
        train_script = train_script,
        imgdir = imgdir,
        epochs = epochs,
        batch_size = batch_size,
        learning_rate = learning_rate,
        momentum = momentum,
        model_dir = model_dir
    )



if __name__ == '__main__':

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
                worker_ports = ((int(exec_conf['myexecutor_1']['WORKER_PORT_1']), int(exec_conf['myexecutor_1']['WORKER_PORT_2']))),
                label = 'myexecutor_1',
                worker_debug = True,             # Default False for shorter logs
                cores_per_worker = int(exec_conf['myexecutor_1']['CORES_PER_WORKER']), # One worker per node
                worker_logdir_root = exec_conf['myexecutor_1']['WORKER_LOGDIR_ROOT'],  #os.getcwd() + '/parsllogs',
                provider = LocalProvider(
                    worker_init = 'source {conda_sh}; conda activate {conda_env}; cd {run_dir}'.format(
                        conda_sh = os.path.join(exec_conf['myexecutor_1']['REMOTE_CONDA_DIR'], 'etc/profile.d/conda.sh'),
                        conda_env = exec_conf['myexecutor_1']['REMOTE_CONDA_ENV'],
                        run_dir = exec_conf['myexecutor_1']['RUN_DIR']
                    ),
                    channel = SSHChannel(
                        hostname = exec_conf['myexecutor_1']['HOST_IP'],
                        username = os.environ['PW_USER'],
                        script_dir = exec_conf['myexecutor_1']['SSH_CHANNEL_SCRIPT_DIR'], # Full path to a script dir where generated scripts could be sent to
                        key_filename = '/home/{PW_USER}/.ssh/pw_id_rsa'.format(PW_USER = os.environ['PW_USER'])
                    )
                )
            ),
            HighThroughputExecutor(
                worker_ports = ((int(exec_conf['myexecutor_2']['WORKER_PORT_1']), int(exec_conf['myexecutor_2']['WORKER_PORT_2']))),
                label = 'myexecutor_2',
                worker_debug = True,             # Default False for shorter logs
                cores_per_worker = int(exec_conf['myexecutor_2']['CORES_PER_WORKER']), # One worker per node
                worker_logdir_root = exec_conf['myexecutor_2']['WORKER_LOGDIR_ROOT'],  #os.getcwd() + '/parsllogs',
                provider = LocalProvider(
                    worker_init = 'source {conda_sh}; conda activate {conda_env}; cd {run_dir}'.format(
                        conda_sh = os.path.join(exec_conf['myexecutor_2']['REMOTE_CONDA_DIR'], 'etc/profile.d/conda.sh'),
                        conda_env = exec_conf['myexecutor_2']['REMOTE_CONDA_ENV'],
                        run_dir = exec_conf['myexecutor_2']['RUN_DIR']
                    ),
                    channel = SSHChannel(
                        hostname = exec_conf['myexecutor_2']['HOST_IP'],
                        username = os.environ['PW_USER'],
                        script_dir = exec_conf['myexecutor_2']['SSH_CHANNEL_SCRIPT_DIR'], # Full path to a script dir where generated scripts could be sent to
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
        './generate_data.py',
        './ships-in-satellite-imagery', args['num_extra_samples'],
        args['max_noise'], args['max_brightness_shift'], args['rotation_range'],
        args['horizontal_flip'], args['vertical_flip'], args['zca_whitening'],
        inputs_dict = {
            "gen_script": {
                "type": "file",
                "global_path": "pw://{cwd}/train_ship_finder/generate_data.py",
                "worker_path": "{remote_dir}/generate_data.py".format(remote_dir = args['remote_dir'])
            },
            "imgdir": {
                "type": "directory",
                "global_path": args['imgdir'],
                "worker_path": "{remote_dir}/ships-in-satellite-imagery".format(remote_dir = args['remote_dir'])
            },
        },
        outputs_dict = {
            "imgdir_gen": {
                "type": "directory",
                "global_path": args['imgdir_out'],
                "worker_path": "{remote_dir}/ships-in-satellite-imagery".format(remote_dir = args['remote_dir'])
            }
        }
    )

    gen_data_fut.result()