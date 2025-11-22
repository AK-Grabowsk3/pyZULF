import numpy as np
import pyvisa as pv
import time

delay = 0.01
batch_delay = 0.01

# Dont know how well this works now, but good foundation
def send_to_Array3400A( Arr, V, freq = 1e3, Vpp = 1.0, Vdc = 0.0 ):

    Arr.query( "*IDN?" )

    Arr.write( "OUTP OFF" )

    Arr.write_ascii_values( "DATA VOLATILE,", V )
    Arr.write( "FUNC:USER VOLATILE" 	)
    Arr.write( f"APPLY:USER {freq:.6e},{Vpp:.6e},{Vdc:.6e}" )

    Arr.write( "BURS:MODE TRIG" )
    Arr.write( "BURS:INT:PER 0.003" )
    
    Arr.write( "BURS:STAT ON" )
    
    Arr.write( "OUTP:SYNC ON" )
    Arr.write( "OUTP ON" )

# Sets up generator with given arbitrary waveform and sample rate
# from Rigol family of products.
def send_to_RigolDG( Rig, V, ch = 1, sampl=1e3 ):

    # Query about current Rigol Digital Generetor model
    dev_name = Rig.query("*IDN?").split(",")[1]
    time.sleep(delay)

    # Turn off output during set up
    Rig.write(f":OUTP{ch} OFF")
    
    # Normalization and data preparation for generator to read.
    # In both cases normalization needs values to be from -1 to 1, 
    # with specific map to binary space different for models
    Vampl = np.max(np.abs(V))

    if Vampl > 0.:
        Vnorm = V / Vampl
    else:
        Vnorm = 0.0
        Vampl = 1.0

    if dev_name in ["DG1062Z", "DG1032Z"]:
        # 14 bit precision
        max_val = (1<<14) - 1

        idle_level = np.uint64( 0.5 * max_val )

        # linear map from [-1,1] to {0,...,16383}:
        Vd = np.array( ( Vnorm + 1 ) / 2 * max_val, dtype = np.uint16 )
        
        # Reverse for Impulse Graph
        Vout = ( 2.0 * Vd / max_val - 1.0 ) * Vampl 

        # Output type
        ftype = "ARB"

    elif dev_name in ["DG952", "DG972", "DG992"]:
        # 16 bit precision
        max_val = (1<<16) - 1
        
        idle_level = np.uint64( max_val )

        # first mapping ]-1,0[ to ]0,1[ and [0,1] to [-1,0]:
        Vnorm -= np.where( Vnorm >= 0.0, 1., -1.00004 )

        # then linear map from [-1,1] to {0,...,65635}:
        Vd = np.array( ( Vnorm + 1 ) / 2 * max_val, dtype = np.uint16 )

        # Reverse for Impulse Graph
        Vout = Vd*2./max_val - 1.
        Vout += np.where( Vout <= 0.0, 1., -1. )
        Vout *= Vampl

        # Output type
        ftype = "SEQ"
    
    # Toggle Burst mode on with external trigger, good to add: 
    # idle level (also dependent on model of generator)
    # for DG900 center, for DG1000Z bottom.
    Rig.write(f":SOUR{ch}:BURS ON")
    Rig.write(f":SOUR{ch}:BURS:MODE TRIG")
    Rig.write(f":SOUR{ch}:BURS:TRIG:SOUR EXT")

    Rig.write(f":SOUR{ch}:BURS:IDLE {idle_level}")
    time.sleep(delay)

    # Set output for arbitrary waveform loaded in volatile memory, max sample rate
    # should be changed to be adapted based on model, some dictionary for example
    Rig.write(f":SOUR{ch}:APPL:{ftype} {min(sampl,60e6):.5e},{Vampl*2:.5e},0.0")
    time.sleep(delay)
    
    Batch_size = 4000
    end_i = 0

    for i in range( len(Vd) // Batch_size - 1 ):
        Rig.write_binary_values( f":SOUR{ch}:DATA:DAC16 VOLATILE,CON,", Vd[i*Batch_size:(i+1)*Batch_size], datatype='H' )
        end_i = Batch_size*(i+1)
        time.sleep(batch_delay)

    Rig.write_binary_values( f":SOUR{ch}:DATA:DAC16 VOLATILE,END,", Vd[end_i:], datatype='H' )
    time.sleep(batch_delay)

    # Double command to turn on, since sometimes after loading last batch generator ignores
    # first instance. It can crash program if after loading waveform a query is send
    Rig.write(f":OUTP{ch} ON")
    Rig.write(f":OUTP{ch} ON")
    time.sleep(delay)

    return Vout

# Queries for error until queue is empty
def Rigol_read_full_error( Rig ):

    # last element is new line character, so we skip it
    mess = Rig.query(":SYST:ERR?")[:-1]
    time.sleep(delay)
    err = ""

    # first character is part of error identification number, 
    # 0 symbolises no error
    while mess[0] != "0":
        err += mess
        mess = Rig.query(":SYST:ERR?")[:-1]
        time.sleep(delay)

    # if no errors than error is set to default "0, no error" message
    if err == "":
        err = mess

    return err

if __name__ == "__main__":
    rm = pv.ResourceManager()
    # takes first device from list ( hopefully Rigol :) )
    dev_addr = rm.list_resources()[0]
    Rig = rm.open_resource( dev_addr )
    print(Rig.query("*IDN?"))

    # testing 50 points send
    y = np.sin( np.linspace(0,2*np.pi), 100 )**2
    send_to_RigolDG( Rig, y )
    Rig.close()
