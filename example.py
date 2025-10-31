import numpy as np
import signal_instructions as si
import pyZULF

example_gates = {
    "sin_strange": si.Shape( 
    lambda x, a=0.0, b=1.0: b * ( np.sin( np.pi * x ) - a * np.sin( np.pi * x * 2 ) / 2.0 ) / (1+np.abs(a)/2),
    duration = 10.0 
),
    "gate_from_file": si.shape_from_file( pyZULF.path_to_pyZULF + "example/dane.dat" )
}

si.load_parameters( new_shapes=example_gates )

if __name__ == "__main__":

    with open( pyZULF.path_to_pyZULF + "example/instructions.txt" ) as file:
        instr = file.read()
    
    signals, schedules = pyZULF.labview_impulse_interface( 
        instructions_txt=instr,
        input_variables=[ 
            ("skew", 3.0),
            ("amp", 10.0)
        ],
        device_list=[("dev",1), ("dev",2)], dt = 0.01 )
    
    commands = si.get_commands( instr )
    impulses, total_duration = si.gather_impulses( commands )
    si.draw_instructions( impulses, signals )