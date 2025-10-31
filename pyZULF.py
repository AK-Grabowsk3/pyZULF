import numpy as np
import matplotlib.pyplot as plt
import pyvisa as pv
import time

import generators as gen
import signal_instructions as si

path_to_pyZULF = ""#"C:\\Users\\ZULF laptop\\Desktop\\ZUL SOFTWARE\\ZULF QSE 2025\\pyZULF\\"

shapes = {}

channels = {
    "x": 0,
    "y": 1,
    "z": 2,
    "s": 3,
    "px": 4,
    "py": 5,
    "pz": 6,
}

calibration = {
    "x": 1.,
    "y": 1.,
    "z": 1.,
    "s": 1.,
    "px": 1.,
    "py": 1.,
    "pz": 1.,
}

si.load_parameters( new_shapes = shapes, new_channels = channels, new_field_scaling=calibration )

def labview_impulse_interface( input_variables, device_list, instructions_txt, dt = 0.01 ):

    si.load_parameters( new_variables = dict( input_variables ) )

    commands = si.get_commands( instructions_txt )
    impulses, total_duration = si.gather_impulses( commands )

    err = ""

    rm = pv.ResourceManager()

    signal_list = []
    time_list = []
    schedule_list = []

    for channel, device in enumerate(device_list):

        schedule = si.get_gate_schedule( impulses, channel )
    
        dev_address = device[0]
        dev_ch = device[1]
        
        # Dev = rm.open_resource( dev_address )

        if len(schedule) > 0:

            t_start = schedule[0]
            duration = schedule[-1]-t_start
        
            t = t_start + np.arange( 1, int(duration/dt) + 1 )*dt
            V = si.calculate_final_voltages( impulses, t, channel )

            # V = gen.send_to_RigolDG( Dev, V, ch=dev_ch, sampl=1e0/dt )
            # err += f", {dev_address}, {dev_ch}, received {len(t)} points, " + gen.Rigol_read_full_error( Dev )

            V = np.insert( V, [0,0], [0.0,0.0] )
            t = np.insert( t, [0,0], [t[0]-dt,t[0]] )
            t[2] += 500e-12


        else:
            # Dev.write(f":OUTP{dev_ch} OFF")

            t = np.array([0.0,total_duration])
            V = np.array([0.0,0.0])

        # Dev.close()

        schedule_list.append( np.array(schedule*1e6,dtype = int) )
        signal_list.append( V )
        time_list.append( t )

    # Obtaining schedules for pulsers
    for channel in range(4,7):
        schedule = si.get_gate_schedule( impulses, channel )
        schedule_list.append( np.array(schedule*1e6,dtype = int) )
    
    signals = list( zip(time_list, signal_list) )
    schedules = [ (el,) for el in schedule_list ]

    # saving any errors gotten from rigol generators to "logs.txt" file
    file = open( path_to_pyZULF + "logs.txt", "a")
    file.write( time.strftime("%H:%M:%S", time.localtime()) + ": " + err + "\n" )
    file.close()

    return signals, schedules

# returning list of tuples with arrays since like 
# that labview won't make them the same size.
# In labview arrays with arrays becomes 2d array, 
# so shorter ones get additional zeros at the end
# python to labview variable transformation:
# tuple                 =>      cluster
# list                  =>      array
# np.ndarray            =>      array
# list{list}            =>      2d array
# list{tuple{arrays}}   =>      array with clusters of arrays
