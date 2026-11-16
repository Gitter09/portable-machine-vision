import hashlib,socket,sys
from pathlib import Path
from types import SimpleNamespace
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools/analysis'))
import portable_network as net


@pytest.mark.parametrize('ip',['192.168.1.30','172.21.30.230','10.42.0.2','169.254.4.8'])
def test_private_link_addresses(ip):assert net.private_address(ip)==ip


@pytest.mark.parametrize('ip',['0.0.0.0','127.0.0.1','8.8.8.8','224.0.0.1','::1','invalid'])
def test_invalid_destinations(ip):
    with pytest.raises(ValueError):net.private_address(ip)


def fake_route(monkeypatch,local='192.168.1.30'):
    class Probe:
        def __enter__(self):return self
        def __exit__(self,*args):pass
        def connect(self,target):assert target==('192.168.1.20',22)
        def getsockname(self):return (local,1234)
        def bind(self,target):
            if target[0]!=local:raise OSError('address not assigned')
    monkeypatch.setattr(net.socket,'socket',lambda *a:Probe())
    monkeypatch.setattr(net.socket,'getaddrinfo',lambda *a,**k:[(socket.AF_INET,socket.SOCK_STREAM,6,'',('192.168.1.20',22))])


def test_receiver_selected_from_route(monkeypatch):
    fake_route(monkeypatch);monkeypatch.setattr(net.sys,'platform','win32')
    assert net.resolve_link('my-board')['receiver_ip']=='192.168.1.30'


def test_nonlocal_receiver_rejected(monkeypatch):
    fake_route(monkeypatch)
    with pytest.raises(OSError):net.resolve_link('my-board','192.168.1.50')


@pytest.mark.parametrize('device,expected',[('en5',True),('en0',False)])
def test_mac_route_not_wifi(monkeypatch,device,expected):
    fake_route(monkeypatch);monkeypatch.setattr(net.sys,'platform','darwin')
    def run(command,**kwargs):
        text='interface: '+device if command[0]=='/sbin/route' else 'Hardware Port: Wi-Fi\nDevice: en0\nHardware Port: USB Ethernet\nDevice: en5\n'
        return SimpleNamespace(stdout=text)
    monkeypatch.setattr(net.subprocess,'run',run)
    if expected:assert net.resolve_link('my-board')['route_interface']=='en5'
    else:
        with pytest.raises(ValueError,match='Wi-Fi'):net.resolve_link('my-board')


def test_ssh_binds_selected_source(monkeypatch):
    fake=object();calls={}
    monkeypatch.setattr(net.socket,'create_connection',lambda *a,**k:calls.update(source=k['source_address']) or fake)
    client=SimpleNamespace(connect=lambda *a,**k:calls.update(sock=k['sock']))
    net.connect_ssh(client,dict(board_ip='192.168.1.20',receiver_ip='192.168.1.30'),'sima')
    assert calls==dict(source=('192.168.1.30',0),sock=fake)


def test_host_key_pin_checked_before_auth():
    client=SimpleNamespace(set_missing_host_key_policy=lambda p:setattr(client,'policy',p))
    key=SimpleNamespace(asbytes=lambda:b'public-test-key');pin=hashlib.sha256(key.asbytes()).hexdigest()
    net.configure_host_keys(client,pin);client.policy.missing_host_key(client,'board',key)
    import paramiko
    with pytest.raises(paramiko.SSHException):client.policy.missing_host_key(client,'board',SimpleNamespace(asbytes=lambda:b'other-key'))
