"""Route-selected private IPv4 links, no network configuration changes."""
import hashlib
import hmac
import ipaddress
import json
import shlex
import socket
import subprocess
import sys

PRIVATE_LINKS=tuple(ipaddress.ip_network(n) for n in ('10.0.0.0/8','172.16.0.0/12','192.168.0.0/16','169.254.0.0/16'))


def private_address(text):
    try:address=ipaddress.IPv4Address(text)
    except ipaddress.AddressValueError:raise ValueError('IPv4 address required') from None
    if not any(address in n for n in PRIVATE_LINKS) or address.is_loopback or address.is_multicast or address.is_unspecified:
        raise ValueError('private or link-local unicast IPv4 required')
    return str(address)


def resolve_link(board_host,receiver_host=None):
    if not board_host or board_host.startswith('<'):raise ValueError('configure the board hostname/IP first')
    addresses=socket.getaddrinfo(board_host,22,family=socket.AF_INET,type=socket.SOCK_STREAM)
    if not addresses:raise ValueError('board has no IPv4 address')
    board_ip=private_address(addresses[0][4][0])
    with socket.socket(socket.AF_INET,socket.SOCK_DGRAM) as probe:
        probe.connect((board_ip,22))  # UDP connect selects a route; sends NO packet.
        automatic=private_address(probe.getsockname()[0])
    receiver=private_address(receiver_host) if receiver_host else automatic
    if receiver==board_ip:raise ValueError('board and receiver cannot have the same address')
    with socket.socket(socket.AF_INET,socket.SOCK_DGRAM) as probe:probe.bind((receiver,0))
    interface=None
    if sys.platform=='darwin':
        r=subprocess.run(['/sbin/route','-n','get',board_ip],capture_output=True,text=True,check=True)
        for line in r.stdout.splitlines():
            if line.strip().startswith('interface:'):interface=line.split(':',1)[1].strip()
        hardware=subprocess.run(['/usr/sbin/networksetup','-listallhardwareports'],capture_output=True,text=True,check=True)
        port=None
        for line in hardware.stdout.splitlines():
            if line.startswith('Hardware Port:'):port=line.split(':',1)[1].strip().lower()
            elif line.startswith('Device:') and line.split(':',1)[1].strip()==interface:
                if port and ('wi-fi' in port or 'wi fi' in port or 'airport' in port):raise ValueError('board route uses Wi-Fi; configure the private Ethernet link first')
        if not interface:raise ValueError('cannot determine Mac route interface')
    return {'board_host':board_host,'board_ip':board_ip,'receiver_ip':receiver,'route_receiver_ip':automatic,'route_interface':interface}


def configure_host_keys(client,expected_sha256=None):
    """Known-hosts or an explicit public-key pin; never silently trust a new host."""
    import paramiko
    if not expected_sha256:
        client.load_system_host_keys();client.set_missing_host_key_policy(paramiko.RejectPolicy());return
    expected=expected_sha256.lower()
    if len(expected)!=64 or any(c not in '0123456789abcdef' for c in expected):raise ValueError('host key pin must be64 hex SHA256 characters')
    class PinnedKey(paramiko.MissingHostKeyPolicy):
        def missing_host_key(self,client,hostname,key):
            actual=hashlib.sha256(key.asbytes()).hexdigest()
            if not hmac.compare_digest(actual,expected):raise paramiko.SSHException('board SSH host-key fingerprint differs')
    client.set_missing_host_key_policy(PinnedKey())


def stop_owned_worker(client,script,output):
    """Signal only this UID's exact Python script AND --output invocation."""
    code='''import json,os,signal,sys
from pathlib import Path
script,output=json.loads(sys.argv[1]);matches=[]
for p in Path('/proc').glob('[0-9]*/cmdline'):
 try:
  argv=p.read_bytes().decode().split(chr(0))
  if p.stat().st_uid!=os.getuid() or not Path(argv[0]).name.startswith('python'):continue
  if script not in argv or '--output' not in argv or argv[argv.index('--output')+1]!=output:continue
  matches.append((int(p.parent.name),argv))
 except (OSError,ValueError,IndexError,UnicodeError):continue
if len(matches)>1:raise SystemExit('ambiguous worker identity; refusing signal')
for pid,argv in matches:
 try:
  if Path('/proc/'+str(pid)+'/cmdline').read_bytes().decode().split(chr(0))!=argv:raise SystemExit('worker identity changed')
  os.kill(pid,signal.SIGINT)
 except (ProcessLookupError,FileNotFoundError):pass
print(json.dumps({'signalled':[p for p,a in matches]}))
'''
    command=shlex.join(['python3','-c',code,json.dumps([script,output])])
    _,stdout,stderr=client.exec_command(command,timeout=15)
    text=stdout.read().decode();error=stderr.read().decode()
    if stdout.channel.recv_exit_status():raise RuntimeError(error or text)
    return json.loads(text)


def connect_ssh(client,link,user,password=None,key_filename=None):
    sock=socket.create_connection((link['board_ip'],22),timeout=15,source_address=(link['receiver_ip'],0))
    try:
        client.connect(link['board_ip'],username=user,password=password,key_filename=key_filename,sock=sock,timeout=15)
    except BaseException:sock.close();raise
