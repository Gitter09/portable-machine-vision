"""Create a NEW host-only environment. Never changes Ethernet, SSH or board SDK."""
import argparse,platform,subprocess,sys
from pathlib import Path


def environment_python(root,system=None):
    return root/('Scripts/python.exe' if (system or platform.system())=='Windows' else 'bin/python')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--package',type=Path,required=True);p.add_argument('--environment',type=Path,required=True)
    p.add_argument('--online',action='store_true',help='explicitly allow PyPI download if preparing dependencies before travel')
    args=p.parse_args();system=platform.system();machine=platform.machine().lower()
    if sys.version_info[:2]!=(3,11):p.error('Use CPython3.11 for this pinned host environment')
    if system not in ('Windows','Darwin'):p.error('This host preparation targets Windows and macOS')
    if system=='Darwin' and machine!='arm64':p.error('Use native Apple Silicon arm64 Python, not a Rosetta/x86 interpreter')
    package=args.package.expanduser().resolve();env=args.environment.expanduser().resolve()
    if env.exists():p.error('Environment already exists; choose a NEW path rather than modifying it')
    subprocess.run([sys.executable,'-B',str(package/'verify.py')],check=True)
    wheels=package/'host-wheels'/('macos-arm64' if system=='Darwin' else 'windows-amd64')
    if not wheels.exists() and system=='Windows':wheels=package/'wheels'
    if not args.online and not wheels.exists():p.error('No matching offline wheels in package; prepare before travel using explicit --online')
    subprocess.run([sys.executable,'-m','venv',str(env)],check=True)
    python=environment_python(env,system)
    command=[str(python),'-m','pip','install','-r',str(package/'requirements.txt')]
    if not args.online:command+=['--no-index','--find-links',str(wheels)]
    subprocess.run(command,check=True);subprocess.run([str(python),'-m','pip','check'],check=True)
    subprocess.run([str(python),'-c',"import av,cv2,numpy,paramiko;av.CodecContext.create('h264','r');print('Host dependencies and H264 decoder loaded')"],check=True)
    print('Host environment ready:',python)
    print('No network settings or board files were changed. Actual Mac/board streaming still requires validation.')


if __name__=='__main__':main()
