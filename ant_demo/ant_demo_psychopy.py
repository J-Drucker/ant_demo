#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This experiment was created using PsychoPy3 Experiment Builder (v2024.1.5),
    on May 23, 2025, at 13:12
If you publish work using this script the most relevant publication is:

    Peirce J, Gray JR, Simpson S, MacAskill M, Höchenberger R, Sogo H, Kastman E, Lindeløv JK. (2019) 
        PsychoPy2: Experiments in behavior made easy Behav Res 51: 195. 
        https://doi.org/10.3758/s13428-018-01193-y

"""

# --- Import packages ---
from psychopy import locale_setup
from psychopy import prefs
from psychopy import plugins
plugins.activatePlugins()
prefs.hardware['audioLib'] = 'ptb'
prefs.hardware['audioLatencyMode'] = '3'
from psychopy import sound, gui, visual, core, data, event, logging, clock, colors, layout, hardware
from psychopy.tools import environmenttools
from psychopy.constants import (NOT_STARTED, STARTED, PLAYING, PAUSED,
                                STOPPED, FINISHED, PRESSED, RELEASED, FOREVER, priority)

import numpy as np  # whole numpy lib is available, prepend 'np.'
from numpy import (sin, cos, tan, log, log10, pi, average,
                   sqrt, std, deg2rad, rad2deg, linspace, asarray)
from numpy.random import random, randint, normal, shuffle, choice as randchoice
import os  # handy system and path functions
import sys  # to get file system encoding

import psychopy.iohub as io
from psychopy.hardware import keyboard

# Run 'Before Experiment' code from code
########################
# Set up Triggering
########################
# Assume no triggers
use_triggers = False

########################
# Set up LSL
########################
try:
    from pylsl import StreamInfo, StreamOutlet
    # Configuration
    STREAM_NAME = "EventMarkers"
    STREAM_TYPE = "Markers"
    NUM_CHANNELS = 1  # Markers are single-channel
    SAMPLING_RATE = 0  # Irregular rate (events are asynchronous)
    CHANNEL_FORMAT = "int32"  # Markers are numbers

    outlet = StreamOutlet(StreamInfo(
        STREAM_NAME, STREAM_TYPE, NUM_CHANNELS, SAMPLING_RATE, CHANNEL_FORMAT, "EventMarkerStream"
        ))
    outlet.push_sample([50]) # start of the experiment
    print('LSL marker stream activated!')
    use_triggers_lsl = True
except Exception as e:
    print(e)
    print('LSL setup failed. Not sending LSL markers')
    use_triggers_lsl = False
    

########################
# Set up Cedrus
########################
import pyxid2

# get a list of all attached XID devices
devices = pyxid2.get_xid_devices()

# If no attached c-pod, don't try to send triggers
try:
    dev = devices[0] # get the first device to use
    print(dev)
    dev.reset_timer()
    dev.set_pulse_duration(210)
    dev.activate_line(bitmask=50) # Start of the experiment
    use_triggers = True
    use_triggers_cpod = True
except IndexError as e:
    print(e)
    print("No Cedrus device found. Script will run without TTL triggers.")
    use_triggers_cpod = False
# This could still fail for some other reason
except Exception as e:
    print(e)
    print("Failure to connect to Cedrus. Script will run without triggers.")
    use_triggers_cpod = False
# --- Setup global variables (available in all functions) ---
# create a device manager to handle hardware (keyboards, mice, mirophones, speakers, etc.)
deviceManager = hardware.DeviceManager()
# ensure that relative paths start from the same directory as this script
_thisDir = os.path.dirname(os.path.abspath(__file__))
# store info about the experiment session
psychopyVersion = '2024.1.5'
expName = 'Oddball_EO-EC'  # from the Builder filename that created this script
# information about this experiment
expInfo = {
    'participant': f"{randint(0, 999999):06.0f}",
    'session': '001',
    'date|hid': data.getDateStr(),
    'expName|hid': expName,
    'psychopyVersion|hid': psychopyVersion,
}

# --- Define some variables which will change depending on pilot mode ---
'''
To run in pilot mode, either use the run/pilot toggle in Builder, Coder and Runner, 
or run the experiment with `--pilot` as an argument. To change what pilot 
#mode does, check out the 'Pilot mode' tab in preferences.
'''
# work out from system args whether we are running in pilot mode
PILOTING = core.setPilotModeFromArgs()
# start off with values from experiment settings
_fullScr = True
_winSize = [1440, 900]
_loggingLevel = logging.getLevel('warning')
# if in pilot mode, apply overrides according to preferences
if PILOTING:
    # force windowed mode
    if prefs.piloting['forceWindowed']:
        _fullScr = False
        # set window size
        _winSize = prefs.piloting['forcedWindowSize']
    # override logging level
    _loggingLevel = logging.getLevel(
        prefs.piloting['pilotLoggingLevel']
    )

def showExpInfoDlg(expInfo):
    """
    Show participant info dialog.
    Parameters
    ==========
    expInfo : dict
        Information about this experiment.
    
    Returns
    ==========
    dict
        Information about this experiment.
    """
    # show participant info dialog
    dlg = gui.DlgFromDict(
        dictionary=expInfo, sortKeys=False, title=expName, alwaysOnTop=True
    )
    if dlg.OK == False:
        core.quit()  # user pressed cancel
    # return expInfo
    return expInfo


def setupData(expInfo, dataDir=None):
    """
    Make an ExperimentHandler to handle trials and saving.
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    dataDir : Path, str or None
        Folder to save the data to, leave as None to create a folder in the current directory.    
    Returns
    ==========
    psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    """
    # remove dialog-specific syntax from expInfo
    for key, val in expInfo.copy().items():
        newKey, _ = data.utils.parsePipeSyntax(key)
        expInfo[newKey] = expInfo.pop(key)
    
    # data file name stem = absolute path + name; later add .psyexp, .csv, .log, etc
    if dataDir is None:
        dataDir = _thisDir
    filename = u'data/%s_%s_%s' % (expInfo['participant'], expName, expInfo['date'])
    # make sure filename is relative to dataDir
    if os.path.isabs(filename):
        dataDir = os.path.commonprefix([dataDir, filename])
        filename = os.path.relpath(filename, dataDir)
    
    # an ExperimentHandler isn't essential but helps with data saving
    thisExp = data.ExperimentHandler(
        name=expName, version='',
        extraInfo=expInfo, runtimeInfo=None,
        originPath='C:\\Users\\neuro\\Desktop\\Demo\\Oddball_EO-EC\\Oddball_EO-EC_lastrun.py',
        savePickle=True, saveWideText=True,
        dataFileName=dataDir + os.sep + filename, sortColumns='time'
    )
    thisExp.setPriority('thisRow.t', priority.CRITICAL)
    thisExp.setPriority('expName', priority.LOW)
    # return experiment handler
    return thisExp


def setupLogging(filename):
    """
    Setup a log file and tell it what level to log at.
    
    Parameters
    ==========
    filename : str or pathlib.Path
        Filename to save log file and data files as, doesn't need an extension.
    
    Returns
    ==========
    psychopy.logging.LogFile
        Text stream to receive inputs from the logging system.
    """
    # this outputs to the screen, not a file
    logging.console.setLevel(_loggingLevel)
    # save a log file for detail verbose info
    logFile = logging.LogFile(filename+'.log', level=_loggingLevel)
    
    return logFile


def setupWindow(expInfo=None, win=None):
    """
    Setup the Window
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    win : psychopy.visual.Window
        Window to setup - leave as None to create a new window.
    
    Returns
    ==========
    psychopy.visual.Window
        Window in which to run this experiment.
    """
    if PILOTING:
        logging.debug('Fullscreen settings ignored as running in pilot mode.')
    
    if win is None:
        # if not given a window to setup, make one
        win = visual.Window(
            size=_winSize, fullscr=_fullScr, screen=0,
            winType='pyglet', allowStencil=False,
            monitor='testMonitor', color=[0,0,0], colorSpace='rgb',
            backgroundImage='', backgroundFit='none',
            blendMode='avg', useFBO=True,
            units='height', 
            checkTiming=False  # we're going to do this ourselves in a moment
        )
    else:
        # if we have a window, just set the attributes which are safe to set
        win.color = [0,0,0]
        win.colorSpace = 'rgb'
        win.backgroundImage = ''
        win.backgroundFit = 'none'
        win.units = 'height'
    win.mouseVisible = False
    win.hideMessage()
    # show a visual indicator if we're in piloting mode
    if PILOTING and prefs.piloting['showPilotingIndicator']:
        win.showPilotingIndicator()
    
    return win


def setupDevices(expInfo, thisExp, win):
    """
    Setup whatever devices are available (mouse, keyboard, speaker, eyetracker, etc.) and add them to 
    the device manager (deviceManager)
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window in which to run this experiment.
    Returns
    ==========
    bool
        True if completed successfully.
    """
    # --- Setup input devices ---
    ioConfig = {}
    
    # Setup iohub keyboard
    ioConfig['Keyboard'] = dict(use_keymap='psychopy')
    
    ioSession = '1'
    if 'session' in expInfo:
        ioSession = str(expInfo['session'])
    ioServer = io.launchHubServer(window=win, **ioConfig)
    # store ioServer object in the device manager
    deviceManager.ioServer = ioServer
    
    # create a default keyboard (e.g. to check for escape)
    if deviceManager.getDevice('defaultKeyboard') is None:
        deviceManager.addDevice(
            deviceClass='keyboard', deviceName='defaultKeyboard', backend='iohub'
        )
    if deviceManager.getDevice('key_instruct_2') is None:
        # initialise key_instruct_2
        key_instruct_2 = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='key_instruct_2',
        )
    if deviceManager.getDevice('key_oddball_response') is None:
        # initialise key_oddball_response
        key_oddball_response = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='key_oddball_response',
        )
    if deviceManager.getDevice('key_instruct') is None:
        # initialise key_instruct
        key_instruct = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='key_instruct',
        )
    # create speaker 'Beep'
    deviceManager.addDevice(
        deviceName='Beep',
        deviceClass='psychopy.hardware.speaker.SpeakerDevice',
        index=4.0
    )
    # return True if completed successfully
    return True

def pauseExperiment(thisExp, win=None, timers=[], playbackComponents=[]):
    """
    Pause this experiment, preventing the flow from advancing to the next routine until resumed.
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window for this experiment.
    timers : list, tuple
        List of timers to reset once pausing is finished.
    playbackComponents : list, tuple
        List of any components with a `pause` method which need to be paused.
    """
    # if we are not paused, do nothing
    if thisExp.status != PAUSED:
        return
    
    # pause any playback components
    for comp in playbackComponents:
        comp.pause()
    # prevent components from auto-drawing
    win.stashAutoDraw()
    # make sure we have a keyboard
    defaultKeyboard = deviceManager.getDevice('defaultKeyboard')
    if defaultKeyboard is None:
        defaultKeyboard = deviceManager.addKeyboard(
            deviceClass='keyboard',
            deviceName='defaultKeyboard',
            backend='ioHub',
        )
    # run a while loop while we wait to unpause
    while thisExp.status == PAUSED:
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=['escape']):
            endExperiment(thisExp, win=win)
        # flip the screen
        win.flip()
    # if stop was requested while paused, quit
    if thisExp.status == FINISHED:
        endExperiment(thisExp, win=win)
    # resume any playback components
    for comp in playbackComponents:
        comp.play()
    # restore auto-drawn components
    win.retrieveAutoDraw()
    # reset any timers
    for timer in timers:
        timer.reset()


def run(expInfo, thisExp, win, globalClock=None, thisSession=None):
    """
    Run the experiment flow.
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    psychopy.visual.Window
        Window in which to run this experiment.
    globalClock : psychopy.core.clock.Clock or None
        Clock to get global time from - supply None to make a new one.
    thisSession : psychopy.session.Session or None
        Handle of the Session object this experiment is being run from, if any.
    """
    # mark experiment as started
    thisExp.status = STARTED
    # make sure variables created by exec are available globally
    exec = environmenttools.setExecEnvironment(globals())
    # get device handles from dict of input devices
    ioServer = deviceManager.ioServer
    # get/create a default keyboard (e.g. to check for escape)
    defaultKeyboard = deviceManager.getDevice('defaultKeyboard')
    if defaultKeyboard is None:
        deviceManager.addDevice(
            deviceClass='keyboard', deviceName='defaultKeyboard', backend='ioHub'
        )
    eyetracker = deviceManager.getDevice('eyetracker')
    # make sure we're running in the directory for this experiment
    os.chdir(_thisDir)
    # get filename from ExperimentHandler for convenience
    filename = thisExp.dataFileName
    frameTolerance = 0.001  # how close to onset before 'same' frame
    endExpNow = False  # flag for 'escape' or other condition => quit the exp
    # get frame duration from frame rate in expInfo
    if 'frameRate' in expInfo and expInfo['frameRate'] is not None:
        frameDur = 1.0 / round(expInfo['frameRate'])
    else:
        frameDur = 1.0 / 60.0  # could not measure, so guess
    
    # Start Code - component code to be run after the window creation
    
    # --- Initialize components for Routine "Oddball_Instruction" ---
    Instruction_Oddball = visual.TextStim(win=win, name='Instruction_Oddball',
        text='You will see a sequence of Os and Xs. Silently count the Xs in your head.\n\n<Press the spacebar to continue>',
        font='Arial',
        units='norm', pos=(0, 0), height=0.1, wrapWidth=1.8, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    key_instruct_2 = keyboard.Keyboard(deviceName='key_instruct_2')
    
    # --- Initialize components for Routine "Countdown" ---
    text_countdown = visual.TextStim(win=win, name='text_countdown',
        text='',
        font='Open Sans',
        pos=(0, 0), height=0.1, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    
    # --- Initialize components for Routine "Oddball_Trial" ---
    Fixation = visual.TextStim(win=win, name='Fixation',
        text='+',
        font='Arial',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    Stimulus = visual.TextStim(win=win, name='Stimulus',
        text=None,
        font='Arial',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    # Run 'Begin Experiment' code from code
    # Stimuli
    import pandas as pd
    import numpy as np
    
    # Are we testing the code out, such that we don't want to require human interaction?
    auto_advance = False
    #auto_advance = True
    
    # Parameters
    n_trials = 10 # trials in a block
    # There are 4 blocks per run, hardcoded
    n_runs = 4 # runs per experiment
    fixation_duration = 1.5
    stimulus_duration = .5
    
    # Create the stimulus set as a 3D array (runs, block, trials)
    stimulus_values = np.empty((n_runs, 4, n_trials), dtype=str)
    for run in range(n_runs):
        for block, n_rare in enumerate(np.random.permutation([1,2,2,3])):
            stimulus_values[run, block, :] = np.random.permutation(np.repeat(['X','O'], [n_rare,n_trials-n_rare]))
    
    # --- Initialize components for Routine "Oddball_Response" ---
    Response_Prompt_Oddball = visual.TextStim(win=win, name='Response_Prompt_Oddball',
        text='How many Xs did you count? Press the appropriate number key.',
        font='Arial',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    key_oddball_response = keyboard.Keyboard(deviceName='key_oddball_response')
    
    # --- Initialize components for Routine "Feedback" ---
    fb = visual.TextStim(win=win, name='fb',
        text='',
        font='Open Sans',
        pos=(0, 0), height=0.1, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    # Set experiment start values for variable component fb_text
    fb_text = ''
    fb_textContainer = []
    
    # --- Initialize components for Routine "Eyes_Closed_Instruction" ---
    Instruction_EC = visual.TextStim(win=win, name='Instruction_EC',
        text='Please close your eyes and relax. When you hear the BEEP, open your eyes again.\n\n<Press the spacebar to continue>',
        font='Arial',
        units='norm', pos=(0, 0), height=0.1, wrapWidth=1.8, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    key_instruct = keyboard.Keyboard(deviceName='key_instruct')
    
    # --- Initialize components for Routine "Eyes_Closed_Condition" ---
    Close_Your_Eyes = visual.TextStim(win=win, name='Close_Your_Eyes',
        text='Close your eyes until you hear the BEEP',
        font='Arial',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    Beep = sound.Sound(
        'A', 
        secs=.3, 
        stereo=True, 
        hamming=True, 
        speaker='Beep',    name='Beep'
    )
    Beep.setVolume(1.0)
    
    # create some handy timers
    
    # global clock to track the time since experiment started
    if globalClock is None:
        # create a clock if not given one
        globalClock = core.Clock()
    if isinstance(globalClock, str):
        # if given a string, make a clock accoridng to it
        if globalClock == 'float':
            # get timestamps as a simple value
            globalClock = core.Clock(format='float')
        elif globalClock == 'iso':
            # get timestamps in ISO format
            globalClock = core.Clock(format='%Y-%m-%d_%H:%M:%S.%f%z')
        else:
            # get timestamps in a custom format
            globalClock = core.Clock(format=globalClock)
    if ioServer is not None:
        ioServer.syncClock(globalClock)
    logging.setDefaultClock(globalClock)
    # routine timer to track time remaining of each (possibly non-slip) routine
    routineTimer = core.Clock()
    win.flip()  # flip window to reset last flip timer
    # store the exact time the global clock started
    expInfo['expStart'] = data.getDateStr(
        format='%Y-%m-%d %Hh%M.%S.%f %z', fractionalSecondDigits=6
    )
    
    # set up handler to look after randomisation of conditions etc
    runs = data.TrialHandler(nReps=n_runs, method='random', 
        extraInfo=expInfo, originPath=-1,
        trialList=[None],
        seed=None, name='runs')
    thisExp.addLoop(runs)  # add the loop to the experiment
    thisRun = runs.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisRun.rgb)
    if thisRun != None:
        for paramName in thisRun:
            globals()[paramName] = thisRun[paramName]
    
    for thisRun in runs:
        currentLoop = runs
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
        )
        # abbreviate parameter names if possible (e.g. rgb = thisRun.rgb)
        if thisRun != None:
            for paramName in thisRun:
                globals()[paramName] = thisRun[paramName]
        
        # set up handler to look after randomisation of conditions etc
        oddball_blocks = data.TrialHandler(nReps=4.0, method='sequential', 
            extraInfo=expInfo, originPath=-1,
            trialList=[None],
            seed=None, name='oddball_blocks')
        thisExp.addLoop(oddball_blocks)  # add the loop to the experiment
        thisOddball_block = oddball_blocks.trialList[0]  # so we can initialise stimuli with some values
        # abbreviate parameter names if possible (e.g. rgb = thisOddball_block.rgb)
        if thisOddball_block != None:
            for paramName in thisOddball_block:
                globals()[paramName] = thisOddball_block[paramName]
        
        for thisOddball_block in oddball_blocks:
            currentLoop = oddball_blocks
            thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[]
            )
            # abbreviate parameter names if possible (e.g. rgb = thisOddball_block.rgb)
            if thisOddball_block != None:
                for paramName in thisOddball_block:
                    globals()[paramName] = thisOddball_block[paramName]
            
            # --- Prepare to start Routine "Oddball_Instruction" ---
            continueRoutine = True
            # update component parameters for each repeat
            thisExp.addData('Oddball_Instruction.started', globalClock.getTime(format='float'))
            # skip this Routine if its 'Skip if' condition is True
            continueRoutine = continueRoutine and not (auto_advance)
            # create starting attributes for key_instruct_2
            key_instruct_2.keys = []
            key_instruct_2.rt = []
            _key_instruct_2_allKeys = []
            # keep track of which components have finished
            Oddball_InstructionComponents = [Instruction_Oddball, key_instruct_2]
            for thisComponent in Oddball_InstructionComponents:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            frameN = -1
            
            # --- Run Routine "Oddball_Instruction" ---
            routineForceEnded = not continueRoutine
            while continueRoutine:
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *Instruction_Oddball* updates
                
                # if Instruction_Oddball is starting this frame...
                if Instruction_Oddball.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    Instruction_Oddball.frameNStart = frameN  # exact frame index
                    Instruction_Oddball.tStart = t  # local t and not account for scr refresh
                    Instruction_Oddball.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(Instruction_Oddball, 'tStartRefresh')  # time at next scr refresh
                    # update status
                    Instruction_Oddball.status = STARTED
                    Instruction_Oddball.setAutoDraw(True)
                
                # if Instruction_Oddball is active this frame...
                if Instruction_Oddball.status == STARTED:
                    # update params
                    pass
                
                # if Instruction_Oddball is stopping this frame...
                if Instruction_Oddball.status == STARTED:
                    if bool(auto_advance):
                        # keep track of stop time/frame for later
                        Instruction_Oddball.tStop = t  # not accounting for scr refresh
                        Instruction_Oddball.tStopRefresh = tThisFlipGlobal  # on global time
                        Instruction_Oddball.frameNStop = frameN  # exact frame index
                        # update status
                        Instruction_Oddball.status = FINISHED
                        Instruction_Oddball.setAutoDraw(False)
                
                # *key_instruct_2* updates
                waitOnFlip = False
                
                # if key_instruct_2 is starting this frame...
                if key_instruct_2.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                    # keep track of start time/frame for later
                    key_instruct_2.frameNStart = frameN  # exact frame index
                    key_instruct_2.tStart = t  # local t and not account for scr refresh
                    key_instruct_2.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(key_instruct_2, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'key_instruct_2.started')
                    # update status
                    key_instruct_2.status = STARTED
                    # keyboard checking is just starting
                    waitOnFlip = True
                    win.callOnFlip(key_instruct_2.clock.reset)  # t=0 on next screen flip
                    win.callOnFlip(key_instruct_2.clearEvents, eventType='keyboard')  # clear events on next screen flip
                if key_instruct_2.status == STARTED and not waitOnFlip:
                    theseKeys = key_instruct_2.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
                    _key_instruct_2_allKeys.extend(theseKeys)
                    if len(_key_instruct_2_allKeys):
                        key_instruct_2.keys = _key_instruct_2_allKeys[0].name  # just the first key pressed
                        key_instruct_2.rt = _key_instruct_2_allKeys[0].rt
                        key_instruct_2.duration = _key_instruct_2_allKeys[0].duration
                        # a response ends the routine
                        continueRoutine = False
                
                # check for quit (typically the Esc key)
                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, win=win)
                    return
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in Oddball_InstructionComponents:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "Oddball_Instruction" ---
            for thisComponent in Oddball_InstructionComponents:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            thisExp.addData('Oddball_Instruction.stopped', globalClock.getTime(format='float'))
            # check responses
            if key_instruct_2.keys in ['', [], None]:  # No response was made
                key_instruct_2.keys = None
            oddball_blocks.addData('key_instruct_2.keys',key_instruct_2.keys)
            if key_instruct_2.keys != None:  # we had a response
                oddball_blocks.addData('key_instruct_2.rt', key_instruct_2.rt)
                oddball_blocks.addData('key_instruct_2.duration', key_instruct_2.duration)
            # the Routine "Oddball_Instruction" was not non-slip safe, so reset the non-slip timer
            routineTimer.reset()
            
            # --- Prepare to start Routine "Countdown" ---
            continueRoutine = True
            # update component parameters for each repeat
            thisExp.addData('Countdown.started', globalClock.getTime(format='float'))
            # skip this Routine if its 'Skip if' condition is True
            continueRoutine = continueRoutine and not (auto_advance)
            # keep track of which components have finished
            CountdownComponents = [text_countdown]
            for thisComponent in CountdownComponents:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            frameN = -1
            
            # --- Run Routine "Countdown" ---
            routineForceEnded = not continueRoutine
            while continueRoutine and routineTimer.getTime() < 3.0:
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *text_countdown* updates
                
                # if text_countdown is starting this frame...
                if text_countdown.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    text_countdown.frameNStart = frameN  # exact frame index
                    text_countdown.tStart = t  # local t and not account for scr refresh
                    text_countdown.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(text_countdown, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'text_countdown.started')
                    # update status
                    text_countdown.status = STARTED
                    text_countdown.setAutoDraw(True)
                
                # if text_countdown is active this frame...
                if text_countdown.status == STARTED:
                    # update params
                    text_countdown.setText(str(3-int(t)), log=False)
                
                # if text_countdown is stopping this frame...
                if text_countdown.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > text_countdown.tStartRefresh + 3-frameTolerance:
                        # keep track of stop time/frame for later
                        text_countdown.tStop = t  # not accounting for scr refresh
                        text_countdown.tStopRefresh = tThisFlipGlobal  # on global time
                        text_countdown.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'text_countdown.stopped')
                        # update status
                        text_countdown.status = FINISHED
                        text_countdown.setAutoDraw(False)
                
                # check for quit (typically the Esc key)
                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, win=win)
                    return
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in CountdownComponents:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "Countdown" ---
            for thisComponent in CountdownComponents:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            thisExp.addData('Countdown.stopped', globalClock.getTime(format='float'))
            # Run 'End Routine' code from EO_begin_marker
            # Send a marker to start the Eyes Open condition
            # Send Eyes Closed begin marker
            if use_triggers:
                if use_triggers_cpod:
                    dev.activate_line(bitmask=3)
                if use_triggers_lsl:
                    outlet.push_sample([3])
            # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
            if routineForceEnded:
                routineTimer.reset()
            else:
                routineTimer.addTime(-3.000000)
            
            # set up handler to look after randomisation of conditions etc
            oddball_trials = data.TrialHandler(nReps=n_trials, method='sequential', 
                extraInfo=expInfo, originPath=-1,
                trialList=[None],
                seed=None, name='oddball_trials')
            thisExp.addLoop(oddball_trials)  # add the loop to the experiment
            thisOddball_trial = oddball_trials.trialList[0]  # so we can initialise stimuli with some values
            # abbreviate parameter names if possible (e.g. rgb = thisOddball_trial.rgb)
            if thisOddball_trial != None:
                for paramName in thisOddball_trial:
                    globals()[paramName] = thisOddball_trial[paramName]
            
            for thisOddball_trial in oddball_trials:
                currentLoop = oddball_trials
                thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
                # pause experiment here if requested
                if thisExp.status == PAUSED:
                    pauseExperiment(
                        thisExp=thisExp, 
                        win=win, 
                        timers=[routineTimer], 
                        playbackComponents=[]
                )
                # abbreviate parameter names if possible (e.g. rgb = thisOddball_trial.rgb)
                if thisOddball_trial != None:
                    for paramName in thisOddball_trial:
                        globals()[paramName] = thisOddball_trial[paramName]
                
                # --- Prepare to start Routine "Oddball_Trial" ---
                continueRoutine = True
                # update component parameters for each repeat
                thisExp.addData('Oddball_Trial.started', globalClock.getTime(format='float'))
                # Run 'Begin Routine' code from code
                # Set the stimulus value from the array created at the beginning of the experiment
                stimulus_value = stimulus_values[
                    runs.thisN, # run 0-3
                    oddball_blocks.thisN, # block 0-3
                    oddball_trials.thisN # trial 0-9
                ]
                
                Stimulus.text = stimulus_value
                
                # Set a flag indicating that the stimulus has not yet been presented
                stim_presented = False
                
                # Set the event code to send with the stimulus. 1 for frequent, 2 for rare
                if use_triggers:
                    if stimulus_value=='O':
                        event_code = 1
                    elif stimulus_value=='X':
                        event_code = 2
                    else:
                        raise
                # keep track of which components have finished
                Oddball_TrialComponents = [Fixation, Stimulus]
                for thisComponent in Oddball_TrialComponents:
                    thisComponent.tStart = None
                    thisComponent.tStop = None
                    thisComponent.tStartRefresh = None
                    thisComponent.tStopRefresh = None
                    if hasattr(thisComponent, 'status'):
                        thisComponent.status = NOT_STARTED
                # reset timers
                t = 0
                _timeToFirstFrame = win.getFutureFlipTime(clock="now")
                frameN = -1
                
                # --- Run Routine "Oddball_Trial" ---
                routineForceEnded = not continueRoutine
                while continueRoutine:
                    # get current time
                    t = routineTimer.getTime()
                    tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                    tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                    frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                    # update/draw components on each frame
                    
                    # *Fixation* updates
                    
                    # if Fixation is starting this frame...
                    if Fixation.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                        # keep track of start time/frame for later
                        Fixation.frameNStart = frameN  # exact frame index
                        Fixation.tStart = t  # local t and not account for scr refresh
                        Fixation.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(Fixation, 'tStartRefresh')  # time at next scr refresh
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'Fixation.started')
                        # update status
                        Fixation.status = STARTED
                        Fixation.setAutoDraw(True)
                    
                    # if Fixation is active this frame...
                    if Fixation.status == STARTED:
                        # update params
                        pass
                    
                    # if Fixation is stopping this frame...
                    if Fixation.status == STARTED:
                        # is it time to stop? (based on global clock, using actual start)
                        if tThisFlipGlobal > Fixation.tStartRefresh + fixation_duration-frameTolerance:
                            # keep track of stop time/frame for later
                            Fixation.tStop = t  # not accounting for scr refresh
                            Fixation.tStopRefresh = tThisFlipGlobal  # on global time
                            Fixation.frameNStop = frameN  # exact frame index
                            # add timestamp to datafile
                            thisExp.timestampOnFlip(win, 'Fixation.stopped')
                            # update status
                            Fixation.status = FINISHED
                            Fixation.setAutoDraw(False)
                    
                    # *Stimulus* updates
                    
                    # if Stimulus is starting this frame...
                    if Stimulus.status == NOT_STARTED and Fixation.status==FINISHED:
                        # keep track of start time/frame for later
                        Stimulus.frameNStart = frameN  # exact frame index
                        Stimulus.tStart = t  # local t and not account for scr refresh
                        Stimulus.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(Stimulus, 'tStartRefresh')  # time at next scr refresh
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'Stimulus.started')
                        # update status
                        Stimulus.status = STARTED
                        Stimulus.setAutoDraw(True)
                    
                    # if Stimulus is active this frame...
                    if Stimulus.status == STARTED:
                        # update params
                        pass
                    
                    # if Stimulus is stopping this frame...
                    if Stimulus.status == STARTED:
                        # is it time to stop? (based on global clock, using actual start)
                        if tThisFlipGlobal > Stimulus.tStartRefresh + stimulus_duration-frameTolerance:
                            # keep track of stop time/frame for later
                            Stimulus.tStop = t  # not accounting for scr refresh
                            Stimulus.tStopRefresh = tThisFlipGlobal  # on global time
                            Stimulus.frameNStop = frameN  # exact frame index
                            # add timestamp to datafile
                            thisExp.timestampOnFlip(win, 'Stimulus.stopped')
                            # update status
                            Stimulus.status = FINISHED
                            Stimulus.setAutoDraw(False)
                    # Run 'Each Frame' code from code
                    # Send an event code with the stimulus. 1 for frequent, 2 for rare
                    if not stim_presented:
                        if Stimulus.status == STARTED:
                            stim_presented = True
                            if use_triggers:
                                if use_triggers_cpod:
                                    dev.activate_line(bitmask=event_code)
                                if use_triggers_lsl:
                                    outlet.push_sample([event_code])
                    
                    # check for quit (typically the Esc key)
                    if defaultKeyboard.getKeys(keyList=["escape"]):
                        thisExp.status = FINISHED
                    if thisExp.status == FINISHED or endExpNow:
                        endExperiment(thisExp, win=win)
                        return
                    
                    # check if all components have finished
                    if not continueRoutine:  # a component has requested a forced-end of Routine
                        routineForceEnded = True
                        break
                    continueRoutine = False  # will revert to True if at least one component still running
                    for thisComponent in Oddball_TrialComponents:
                        if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                            continueRoutine = True
                            break  # at least one component has not yet finished
                    
                    # refresh the screen
                    if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                        win.flip()
                
                # --- Ending Routine "Oddball_Trial" ---
                for thisComponent in Oddball_TrialComponents:
                    if hasattr(thisComponent, "setAutoDraw"):
                        thisComponent.setAutoDraw(False)
                thisExp.addData('Oddball_Trial.stopped', globalClock.getTime(format='float'))
                # the Routine "Oddball_Trial" was not non-slip safe, so reset the non-slip timer
                routineTimer.reset()
                thisExp.nextEntry()
                
                if thisSession is not None:
                    # if running in a Session with a Liaison client, send data up to now
                    thisSession.sendExperimentData()
            # completed n_trials repeats of 'oddball_trials'
            
            
            # --- Prepare to start Routine "Oddball_Response" ---
            continueRoutine = True
            # update component parameters for each repeat
            thisExp.addData('Oddball_Response.started', globalClock.getTime(format='float'))
            # skip this Routine if its 'Skip if' condition is True
            continueRoutine = continueRoutine and not (auto_advance)
            # create starting attributes for key_oddball_response
            key_oddball_response.keys = []
            key_oddball_response.rt = []
            _key_oddball_response_allKeys = []
            # Run 'Begin Routine' code from Eyes_Open_end_marker
            # Send Eyes Open end marker
            if use_triggers:
                if use_triggers_cpod:
                    dev.activate_line(bitmask=4)
                if use_triggers_lsl:
                    outlet.push_sample([4])
                
            # keep track of which components have finished
            Oddball_ResponseComponents = [Response_Prompt_Oddball, key_oddball_response]
            for thisComponent in Oddball_ResponseComponents:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            frameN = -1
            
            # --- Run Routine "Oddball_Response" ---
            routineForceEnded = not continueRoutine
            while continueRoutine:
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *Response_Prompt_Oddball* updates
                
                # if Response_Prompt_Oddball is starting this frame...
                if Response_Prompt_Oddball.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    Response_Prompt_Oddball.frameNStart = frameN  # exact frame index
                    Response_Prompt_Oddball.tStart = t  # local t and not account for scr refresh
                    Response_Prompt_Oddball.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(Response_Prompt_Oddball, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'Response_Prompt_Oddball.started')
                    # update status
                    Response_Prompt_Oddball.status = STARTED
                    Response_Prompt_Oddball.setAutoDraw(True)
                
                # if Response_Prompt_Oddball is active this frame...
                if Response_Prompt_Oddball.status == STARTED:
                    # update params
                    pass
                
                # *key_oddball_response* updates
                waitOnFlip = False
                
                # if key_oddball_response is starting this frame...
                if key_oddball_response.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    key_oddball_response.frameNStart = frameN  # exact frame index
                    key_oddball_response.tStart = t  # local t and not account for scr refresh
                    key_oddball_response.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(key_oddball_response, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'key_oddball_response.started')
                    # update status
                    key_oddball_response.status = STARTED
                    # keyboard checking is just starting
                    waitOnFlip = True
                    win.callOnFlip(key_oddball_response.clock.reset)  # t=0 on next screen flip
                    win.callOnFlip(key_oddball_response.clearEvents, eventType='keyboard')  # clear events on next screen flip
                if key_oddball_response.status == STARTED and not waitOnFlip:
                    theseKeys = key_oddball_response.getKeys(keyList=['1','2','3','4','5','6','7','8','9','0'], ignoreKeys=["escape"], waitRelease=False)
                    _key_oddball_response_allKeys.extend(theseKeys)
                    if len(_key_oddball_response_allKeys):
                        key_oddball_response.keys = _key_oddball_response_allKeys[-1].name  # just the last key pressed
                        key_oddball_response.rt = _key_oddball_response_allKeys[-1].rt
                        key_oddball_response.duration = _key_oddball_response_allKeys[-1].duration
                        # was this correct?
                        if (key_oddball_response.keys == str(str(np.sum(stimulus_values[runs.thisN,oddball_blocks.thisN,:]=='X')))) or (key_oddball_response.keys == str(np.sum(stimulus_values[runs.thisN,oddball_blocks.thisN,:]=='X'))):
                            key_oddball_response.corr = 1
                        else:
                            key_oddball_response.corr = 0
                        # a response ends the routine
                        continueRoutine = False
                
                # check for quit (typically the Esc key)
                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, win=win)
                    return
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in Oddball_ResponseComponents:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "Oddball_Response" ---
            for thisComponent in Oddball_ResponseComponents:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            thisExp.addData('Oddball_Response.stopped', globalClock.getTime(format='float'))
            # check responses
            if key_oddball_response.keys in ['', [], None]:  # No response was made
                key_oddball_response.keys = None
                # was no response the correct answer?!
                if str(str(np.sum(stimulus_values[runs.thisN,oddball_blocks.thisN,:]=='X'))).lower() == 'none':
                   key_oddball_response.corr = 1;  # correct non-response
                else:
                   key_oddball_response.corr = 0;  # failed to respond (incorrectly)
            # store data for oddball_blocks (TrialHandler)
            oddball_blocks.addData('key_oddball_response.keys',key_oddball_response.keys)
            oddball_blocks.addData('key_oddball_response.corr', key_oddball_response.corr)
            if key_oddball_response.keys != None:  # we had a response
                oddball_blocks.addData('key_oddball_response.rt', key_oddball_response.rt)
                oddball_blocks.addData('key_oddball_response.duration', key_oddball_response.duration)
            # Run 'End Routine' code from Eyes_Open_end_marker
            ## Send an event trigger corresponding with the participant response
            #try:
            #    if key_oddball_response.corr:
            #        fb_text = 'Correct!'
            #        fb_col = 'limegreen'
            #    else:
            #        fb_text = 'Incorrect'
            #        fb_col = 'red'
            #
            #
            #if use_triggers:
            #    if use_triggers_cpod:
            #        dev.activate_line(bitmask=4)
            #    if use_triggers_lsl:
            #        outlet.push_sample([4])
            # the Routine "Oddball_Response" was not non-slip safe, so reset the non-slip timer
            routineTimer.reset()
            
            # --- Prepare to start Routine "Feedback" ---
            continueRoutine = True
            # update component parameters for each repeat
            thisExp.addData('Feedback.started', globalClock.getTime(format='float'))
            # skip this Routine if its 'Skip if' condition is True
            continueRoutine = continueRoutine and not (auto_advance)
            # Run 'Begin Routine' code from fb_code
            # Check if the key press was correct or not.
            # This routine will need to follow another routine with a 
            # key response component in it called "key_resp" 
            # and the "store correct" option enabled. 
            # If your experiment is missing that you will 
            # not receive feedback and an error message will be displayed.
            
            # If a key response component has been added and feedback is functioning.
            # 1. remove lines 12, 13, 15, 22 and 23.
            # 2. dedent lines 16 to 21
            
            fb_text = 'no key_resp component found - look at the Std out window for info'
            fb_col = 'black'
            
            try:
                if key_oddball_response.corr:
                    fb_text = 'Correct!'
                    fb_col = 'limegreen'
                else:
                    fb_text = 'Incorrect'
                    fb_col = 'red'
            except:
                print('Make sure that you have:\n1. a routine with a keyboard component in it called "key_resp"\n 2. In the key_Resp component in the "data" tab select "Store Correct".\n in the "Correct answer" field use "$corrAns" (where corrAns is a column header in your conditions file indicating the correct key press')
            
            fb.setColor(fb_col, colorSpace='rgb')
            fb.setText(fb_text)
            # keep track of which components have finished
            FeedbackComponents = [fb]
            for thisComponent in FeedbackComponents:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            frameN = -1
            
            # --- Run Routine "Feedback" ---
            routineForceEnded = not continueRoutine
            while continueRoutine:
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *fb* updates
                
                # if fb is starting this frame...
                if fb.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    fb.frameNStart = frameN  # exact frame index
                    fb.tStart = t  # local t and not account for scr refresh
                    fb.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(fb, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'fb.started')
                    # update status
                    fb.status = STARTED
                    fb.setAutoDraw(True)
                
                # if fb is active this frame...
                if fb.status == STARTED:
                    # update params
                    pass
                
                # if fb is stopping this frame...
                if fb.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > fb.tStartRefresh + 1.0-frameTolerance:
                        # keep track of stop time/frame for later
                        fb.tStop = t  # not accounting for scr refresh
                        fb.tStopRefresh = tThisFlipGlobal  # on global time
                        fb.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'fb.stopped')
                        # update status
                        fb.status = FINISHED
                        fb.setAutoDraw(False)
                
                # check for quit (typically the Esc key)
                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, win=win)
                    return
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in FeedbackComponents:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "Feedback" ---
            for thisComponent in FeedbackComponents:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            thisExp.addData('Feedback.stopped', globalClock.getTime(format='float'))
            
            # the Routine "Feedback" was not non-slip safe, so reset the non-slip timer
            routineTimer.reset()
            thisExp.nextEntry()
            
            if thisSession is not None:
                # if running in a Session with a Liaison client, send data up to now
                thisSession.sendExperimentData()
        # completed 4.0 repeats of 'oddball_blocks'
        
        
        # --- Prepare to start Routine "Eyes_Closed_Instruction" ---
        continueRoutine = True
        # update component parameters for each repeat
        thisExp.addData('Eyes_Closed_Instruction.started', globalClock.getTime(format='float'))
        # skip this Routine if its 'Skip if' condition is True
        continueRoutine = continueRoutine and not (auto_advance)
        # create starting attributes for key_instruct
        key_instruct.keys = []
        key_instruct.rt = []
        _key_instruct_allKeys = []
        # keep track of which components have finished
        Eyes_Closed_InstructionComponents = [Instruction_EC, key_instruct]
        for thisComponent in Eyes_Closed_InstructionComponents:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "Eyes_Closed_Instruction" ---
        routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *Instruction_EC* updates
            
            # if Instruction_EC is starting this frame...
            if Instruction_EC.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                Instruction_EC.frameNStart = frameN  # exact frame index
                Instruction_EC.tStart = t  # local t and not account for scr refresh
                Instruction_EC.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(Instruction_EC, 'tStartRefresh')  # time at next scr refresh
                # update status
                Instruction_EC.status = STARTED
                Instruction_EC.setAutoDraw(True)
            
            # if Instruction_EC is active this frame...
            if Instruction_EC.status == STARTED:
                # update params
                pass
            
            # if Instruction_EC is stopping this frame...
            if Instruction_EC.status == STARTED:
                if bool(auto_advance):
                    # keep track of stop time/frame for later
                    Instruction_EC.tStop = t  # not accounting for scr refresh
                    Instruction_EC.tStopRefresh = tThisFlipGlobal  # on global time
                    Instruction_EC.frameNStop = frameN  # exact frame index
                    # update status
                    Instruction_EC.status = FINISHED
                    Instruction_EC.setAutoDraw(False)
            
            # *key_instruct* updates
            waitOnFlip = False
            
            # if key_instruct is starting this frame...
            if key_instruct.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                key_instruct.frameNStart = frameN  # exact frame index
                key_instruct.tStart = t  # local t and not account for scr refresh
                key_instruct.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(key_instruct, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'key_instruct.started')
                # update status
                key_instruct.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(key_instruct.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(key_instruct.clearEvents, eventType='keyboard')  # clear events on next screen flip
            
            # if key_instruct is stopping this frame...
            if key_instruct.status == STARTED:
                if bool(auto_advance):
                    # keep track of stop time/frame for later
                    key_instruct.tStop = t  # not accounting for scr refresh
                    key_instruct.tStopRefresh = tThisFlipGlobal  # on global time
                    key_instruct.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'key_instruct.stopped')
                    # update status
                    key_instruct.status = FINISHED
                    key_instruct.status = FINISHED
            if key_instruct.status == STARTED and not waitOnFlip:
                theseKeys = key_instruct.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
                _key_instruct_allKeys.extend(theseKeys)
                if len(_key_instruct_allKeys):
                    key_instruct.keys = _key_instruct_allKeys[0].name  # just the first key pressed
                    key_instruct.rt = _key_instruct_allKeys[0].rt
                    key_instruct.duration = _key_instruct_allKeys[0].duration
                    # a response ends the routine
                    continueRoutine = False
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in Eyes_Closed_InstructionComponents:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "Eyes_Closed_Instruction" ---
        for thisComponent in Eyes_Closed_InstructionComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        thisExp.addData('Eyes_Closed_Instruction.stopped', globalClock.getTime(format='float'))
        # check responses
        if key_instruct.keys in ['', [], None]:  # No response was made
            key_instruct.keys = None
        runs.addData('key_instruct.keys',key_instruct.keys)
        if key_instruct.keys != None:  # we had a response
            runs.addData('key_instruct.rt', key_instruct.rt)
            runs.addData('key_instruct.duration', key_instruct.duration)
        # the Routine "Eyes_Closed_Instruction" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # --- Prepare to start Routine "Eyes_Closed_Condition" ---
        continueRoutine = True
        # update component parameters for each repeat
        thisExp.addData('Eyes_Closed_Condition.started', globalClock.getTime(format='float'))
        Beep.setSound('A', secs=.3, hamming=True)
        Beep.setVolume(1.0, log=False)
        Beep.seek(0)
        # Run 'Begin Routine' code from EC_markers
        # Send Eyes Closed begin marker
        if use_triggers:
            dev.activate_line(bitmask=5)
        
        # Set a flag indicating that the beep hasn't happened
        beep_presented = False
        # keep track of which components have finished
        Eyes_Closed_ConditionComponents = [Close_Your_Eyes, Beep]
        for thisComponent in Eyes_Closed_ConditionComponents:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "Eyes_Closed_Condition" ---
        routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *Close_Your_Eyes* updates
            
            # if Close_Your_Eyes is starting this frame...
            if Close_Your_Eyes.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                Close_Your_Eyes.frameNStart = frameN  # exact frame index
                Close_Your_Eyes.tStart = t  # local t and not account for scr refresh
                Close_Your_Eyes.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(Close_Your_Eyes, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'Close_Your_Eyes.started')
                # update status
                Close_Your_Eyes.status = STARTED
                Close_Your_Eyes.setAutoDraw(True)
            
            # if Close_Your_Eyes is active this frame...
            if Close_Your_Eyes.status == STARTED:
                # update params
                pass
            
            # if Close_Your_Eyes is stopping this frame...
            if Close_Your_Eyes.status == STARTED:
                if bool(Beep.status==FINISHED):
                    # keep track of stop time/frame for later
                    Close_Your_Eyes.tStop = t  # not accounting for scr refresh
                    Close_Your_Eyes.tStopRefresh = tThisFlipGlobal  # on global time
                    Close_Your_Eyes.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'Close_Your_Eyes.stopped')
                    # update status
                    Close_Your_Eyes.status = FINISHED
                    Close_Your_Eyes.setAutoDraw(False)
            
            # if Beep is starting this frame...
            if Beep.status == NOT_STARTED and t >= 20-frameTolerance:
                # keep track of start time/frame for later
                Beep.frameNStart = frameN  # exact frame index
                Beep.tStart = t  # local t and not account for scr refresh
                Beep.tStartRefresh = tThisFlipGlobal  # on global time
                # add timestamp to datafile
                thisExp.addData('Beep.started', t)
                # update status
                Beep.status = STARTED
                Beep.play()  # start the sound (it finishes automatically)
            
            # if Beep is stopping this frame...
            if Beep.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > Beep.tStartRefresh + .3-frameTolerance:
                    # keep track of stop time/frame for later
                    Beep.tStop = t  # not accounting for scr refresh
                    Beep.tStopRefresh = tThisFlipGlobal  # on global time
                    Beep.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.addData('Beep.stopped', t)
                    # update status
                    Beep.status = FINISHED
                    Beep.stop()
            # update Beep status according to whether it's playing
            if Beep.isPlaying:
                Beep.status = STARTED
            elif Beep.isFinished:
                Beep.status = FINISHED
            # Run 'Each Frame' code from EC_markers
            # Send Eyes Closed end marker (6)
            if not beep_presented:
                if Beep.status == STARTED:
                    if use_triggers:
                        beep_presented = True
                        dev.activate_line(bitmask=6)
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in Eyes_Closed_ConditionComponents:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "Eyes_Closed_Condition" ---
        for thisComponent in Eyes_Closed_ConditionComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        thisExp.addData('Eyes_Closed_Condition.stopped', globalClock.getTime(format='float'))
        # the Routine "Eyes_Closed_Condition" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        thisExp.nextEntry()
        
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
    # completed n_runs repeats of 'runs'
    
    # Run 'End Experiment' code from code
    # Marker to end the experiment
    if use_triggers:
        if use_triggers_cpod:
            dev.activate_line(bitmask=51) # End of the experiment
        if use_triggers_lsl:
            outlet.push_sample([51]) # end of the experiment
    
    
    
    
    # mark experiment as finished
    endExperiment(thisExp, win=win)


def saveData(thisExp):
    """
    Save data from this experiment
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    """
    filename = thisExp.dataFileName
    # these shouldn't be strictly necessary (should auto-save)
    thisExp.saveAsWideText(filename + '.csv', delim='auto')
    thisExp.saveAsPickle(filename)


def endExperiment(thisExp, win=None):
    """
    End this experiment, performing final shut down operations.
    
    This function does NOT close the window or end the Python process - use `quit` for this.
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window for this experiment.
    """
    if win is not None:
        # remove autodraw from all current components
        win.clearAutoDraw()
        # Flip one final time so any remaining win.callOnFlip() 
        # and win.timeOnFlip() tasks get executed
        win.flip()
    # mark experiment handler as finished
    thisExp.status = FINISHED
    # shut down eyetracker, if there is one
    if deviceManager.getDevice('eyetracker') is not None:
        deviceManager.removeDevice('eyetracker')
    logging.flush()


def quit(thisExp, win=None, thisSession=None):
    """
    Fully quit, closing the window and ending the Python process.
    
    Parameters
    ==========
    win : psychopy.visual.Window
        Window to close.
    thisSession : psychopy.session.Session or None
        Handle of the Session object this experiment is being run from, if any.
    """
    thisExp.abort()  # or data files will save again on exit
    # make sure everything is closed down
    if win is not None:
        # Flip one final time so any remaining win.callOnFlip() 
        # and win.timeOnFlip() tasks get executed before quitting
        win.flip()
        win.close()
    # shut down eyetracker, if there is one
    if deviceManager.getDevice('eyetracker') is not None:
        deviceManager.removeDevice('eyetracker')
    logging.flush()
    if thisSession is not None:
        thisSession.stop()
    # terminate Python process
    core.quit()


# if running this experiment as a script...
if __name__ == '__main__':
    # call all functions in order
    thisExp = setupData(expInfo=expInfo)
    logFile = setupLogging(filename=thisExp.dataFileName)
    win = setupWindow(expInfo=expInfo)
    setupDevices(expInfo=expInfo, thisExp=thisExp, win=win)
    run(
        expInfo=expInfo, 
        thisExp=thisExp, 
        win=win,
        globalClock='float'
    )
    saveData(thisExp=thisExp)
    quit(thisExp=thisExp, win=win)
