from pycaw.pycaw import AudioUtilities, ISimpleAudioVolume
import spotipy
from spotipy.oauth2 import SpotifyOAuth
from dotenv import load_dotenv
import os

load_dotenv('.env')


def mute_audio():
    """Mutes all audio on the system"""
    sessions = AudioUtilities.GetAllSessions()
    for session in sessions:
        interface = session._ctl.QueryInterface(ISimpleAudioVolume)
        interface.SetMasterVolume(0, None)


def unmute_audio():
    """Unmutes all audio on the system"""
    sessions = AudioUtilities.GetAllSessions()
    for session in sessions:
        interface = session._ctl.QueryInterface(ISimpleAudioVolume)
        interface.SetMasterVolume(1, None)  


def set_full_volume():
    """Sets all audio on the system to full volume"""
    sessions = AudioUtilities.GetAllSessions()
    for session in sessions:
        interface = session._ctl.QueryInterface(ISimpleAudioVolume)
        interface.SetMasterVolume(1, None)

def set_half_volume():
    """Sets all audio on the system to half volume"""
    sessions = AudioUtilities.GetAllSessions()
    for session in sessions:
        interface = session._ctl.QueryInterface(ISimpleAudioVolume)
        interface.SetMasterVolume(0.5, None)

def set_qtr_volume():
    """Sets all audio on the system to quarter volume"""
    sessions = AudioUtilities.GetAllSessions()
    for session in sessions:
        interface = session._ctl.QueryInterface(ISimpleAudioVolume)
        interface.SetMasterVolume(0.25, None)



