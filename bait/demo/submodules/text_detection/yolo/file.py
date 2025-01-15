from os import makedirs, environ
from os.path import abspath, dirname, exists, join
from subprocess import run as run_subproc, CalledProcessError


def get_weights(
    weight_path: str,
    weight_server: str = "ts0107@192.168.1.41",
    out_dir: str = "./weights",
) -> str:
    """
    Downloads a weights file from a remote server using rsync.

    Args:
        weight_path: The path to the weights file on the remote server.
        weight_server: The IP address and username of the remote server.
        out_dir: The local directory to download the weights file to.

    Returns:
        The path to the downloaded weights file on the local machine.

    Raises:
        CalledProcessError: If the rsync command fails.
    """
    weight_name = weight_path.split("/")[-1]
    abs_out_dir = abspath(out_dir)
    dst_file = join(abs_out_dir, weight_name)

    if not exists(out_dir):
        makedirs(abs_out_dir)

    if not exists(dst_file):
        command = "rsync -aP {}:{} {}".format(weight_server, weight_path, out_dir)
        try:
            print("Downloading weights ...")
            run_subproc(command, shell=True)
        except CalledProcessError as e:
            raise str(e)
    else:
        print("Weights is already located. No downloading.")

    return dst_file
