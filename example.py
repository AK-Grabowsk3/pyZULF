import numpy as np
import signal_instructions as si

example_shapes = {
    "sin_strange": si.Shape( 
    lambda x, a=0.0: ( np.sin( np.pi * x ) - a * np.sin( np.pi * x * 2 ) / 2.0 ) / (1+np.abs(a)/2),
    duration = 10.0 
),
    "gate_from_file": si.shape_from_file( "example/dane.dat" )
}

channels = { "x": 0, "y": 1 }

calibration = { "x": 1., "y": 1. }

input_variables={ "skew": 3.0, "amp": 10.0 }

si.load_parameters( 
    new_shapes = example_shapes, 
    new_channels = channels, 
    new_field_scaling=calibration,
    new_variables=input_variables
)

if __name__ == "__main__":

    file = open( "example/instructions.txt" )
    instr = file.read()
    file.close()

    dt = 0.01
    
    si.load_parameters( new_variables = input_variables )

    commands = si.get_commands_from_instructions( instr )
    impulses, total_duration = si.gather_impulses( commands )

    signal_list = []
    time_list = []

    for ch in [0,1]:

        schedule = si.get_gate_schedule( impulses, ch )
        if len(schedule) == 0:
            
            signal_list.append( [0,0] )
            time_list.append( [0,total_duration] )

            continue

        t_start = schedule[0]
        duration = schedule[-1]-t_start

        t = t_start + np.arange( 1, int(duration/dt) + 1 )*dt
        V = si.calculate_final_voltages( impulses, t, ch )

        V = np.insert( V, [0,0], [0.0,0.0] )
        t = np.insert( t, [0,0], [t[0]-dt,t[0]] )
        t[2] += 500e-12
        
        signal_list.append( V )
        time_list.append( t )

    signals = list( zip(time_list, signal_list) )
    
    si.draw_instructions( impulses, signals )