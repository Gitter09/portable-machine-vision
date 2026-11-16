"""Clone-only host setup: all assets/wheels and the new environment stay in repo."""
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parent
if __name__=='__main__':
    subprocess.run([sys.executable,str(ROOT/'tools/setup_portable_host.py'),'--package',str(ROOT/'deployment/portable-v1'),'--environment',str(ROOT/'local/host-env'),*sys.argv[1:]],check=True)
