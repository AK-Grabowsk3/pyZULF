import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline, interp1d


axis_impulse_sep = "<-"
modif_sep = ","
modif_set_chr = ":"
param_sep = ","
signal_variable = "signal_base_value"


# Class for storing signal shape as a creating stone of gates.
# Also used as general curve shape for experiment setup like guiding 
# field shape etc.
class Shape:
    def __init__( self, curve, duration = 1.0, duration_f = None ):
        self.T = duration 
        self.Tf = duration_f 
        self.V = curve

    def get_duration( self, par=None ):
        if (par is None) or (self.Tf is None):
            return self.T
        else:
            return self.Tf( *par )

    def get_curve( self, x, par=None ):
        if par is None:
            return self.V(x)
        else:
            return self.V(x, *par)

# dictionary for predefined shapes
shapes_dict:dict[str:Shape] = {
	"sin": Shape( 
    lambda x, A=1.0, phi=0.0, N_cycl=1.0: A*np.sin( N_cycl * x * 2 * np.pi + phi*np.pi ), 
),
	"line": Shape(
    lambda x, V=0.0, dV=0.0: V+dV*x,
),
	"exp": Shape( 
    lambda x, dx=1.0, V=0.0, dV=0.0: V + dV * ( 1 - np.exp(x/dx) ) / ( 1 - np.exp(1/dx) ),
)
}

# dictionary for numbering axis
channels:dict[str,int] = {}
# dictionary with custom variables
variables:dict[str,float] = {"get_current_time()": 0.0}
# dictionary with coil calibration
field_scaling:dict[str,float] = {}

# scaling factor to use proper angle unit. 
# 1 for radians, pi/180 for degrees
angle_scaling = np.pi/180

# Class that is a building block for signal. It only represents 
# a single channel signal shape represented by a single gate and 
# for given parameters. When V is called it will give 0 when 
# outside of its duration or not proper channel. When inside 
# value is properly calculated. 
class Impulse:
    def __init__( self, t_start, duration, channel, sname, param, scale, expr ):
        self.t_i = t_start
        self.t_f = t_start + duration
        self.T = duration
        self.ch:int = channel
        self.p = param
        self.sname = sname
        self.g:Shape = shapes_dict[sname]
        self.e = expr
        self.s = scale
        
        global variables

        mod_func = eval( "lambda " + signal_variable + ": " + signal_variable + " " + expr, variables )

        def V_singular( t, ch=0 ):
            if (t > self.t_i) and (t <= self.t_f) and (ch == self.ch):
                return mod_func( self.g.get_curve( ( t - t_start )/duration, param ) )*self.s
            else:
                return 0.0
        
        self.V = np.vectorize( V_singular, excluded=[1] )

    def __call__( self, t, ch ):
        return self.V(t,ch)

# function to set global parameters, since they don't easily transfer file to file
def load_parameters( new_variables = None, new_shapes = None, new_channels = None, new_field_scaling=None ):
    global variables
    global channels
    global shapes_dict
    global field_scaling

    if not(new_variables is None):
        variables.update( new_variables )

    if not(new_field_scaling is None):
        field_scaling.update( new_field_scaling )
    if not(new_shapes is None):
        shapes_dict.update( new_shapes )
    if not(new_channels is None):
        channels.update( new_channels )

    pass

# Allows to read file with two columns, time and field/voltage, to 
# create shape element representing it. It uses CubicSpline function
# from scipy to obtain interpolation based on data from file. By 
# default boundary condition is set to "natural", meaning at the 
# edges second derivative is set to 0. 
def shape_from_file_cubic( fname, boundary_condition = "natural" ):
    data = np.loadtxt( fname )

    time = data[:,0]
    signal = data[:,1]

    time = time - time[0]
    duration = time[-1]

    func = lambda x, s=1.0: CubicSpline( time/duration, signal, bc_type=boundary_condition )(x)*s

    return Shape( curve = func, duration = duration )

# Allows to read file with two columns, time and field/voltage, to 
# create shape element representing it. It uses interp1 function
# from scipy to obtain interpolation based on data from file. By 
# default kind of interpolation is set to "linear". For more 
# information go to scipy.interpolate.interp1 documentation.
def shape_from_file( fname, interp_type = "linear" ):
    data = np.loadtxt( fname )

    time = data[:,0]
    signal = data[:,1]

    time = time - time[0]
    duration = time[-1]

    func = lambda x, s=1.0: interp1d( time/duration, signal, kind = interp_type )(x)*s

    return Shape( curve = func, duration = duration )

# Reads string "instructions_txt" with raw instruction lines 
# and cleans it from comments (text after "#" character) and 
# ommits empty lines
def get_commands( instructions_txt ):
    
    lines = instructions_txt.splitlines()
    commands = []
    for line in lines:
        end = line.find("#")
        if end >= 0:
            command = line[:end].strip()
        else:
            command = line.strip()

        if command != "":
            commands.append( command )

    return commands

# decompose_command(line) splits command given as 
# string "line" variable into four parts:
# - shape name
# - general modifiers from square brackets
# - shape specific parameters from normal brackets
# - axis indicator from right side of ">" symbol
# It does not converted variables to values.
def decompose_command( line ):

    m = line.find( axis_impulse_sep )

    command_txt = line[m+len(axis_impulse_sep):].strip()
    axis = line[:m].strip() 

    # find square 
    modif = []

    beg = command_txt.find("[")
    end = command_txt.find("]")
    if ( beg >= 0 ) and ( end > 0 ):
        
        elements = command_txt[beg+len(modif_sep):end].split( sep = modif_sep )
        elements = [ el.split( sep = modif_set_chr ) for el in elements ]
        modif = [ ( el[0].strip(), read_param(el[1]) ) for el in elements ]

        command_txt = command_txt[end+1:]
    
    elif (end <= 0)^(beg < 0) :
        raise Exception( f"Square brackets not closed in line:\n\t{line}" )
    
    for sh_name in sorted(shapes_dict.keys(), key=lambda x: len(x), reverse=True):
        sh_name_loc = command_txt.find(sh_name)
        
        if sh_name_loc >= 0:
            sname = sh_name
            break
    
    if sh_name_loc < 0:
        raise Exception( f"In line: \n{line}\n could not find any existing shape names." )

    command_txt = command_txt[ sh_name_loc + len(sname) :]

    beg = command_txt.find("(")
    
    end = -1 #command_txt.find(")")
    
    opened_br = 1

    for curr in range(beg+1, len(command_txt)):

        if command_txt[curr] == "(":
            opened_br += 1
        if command_txt[curr] == ")":
            opened_br -= 1

        if opened_br == 0:
            end = curr
            break
    
    param = None

    if ( beg >= 0 ) and ( end > 0 ):

        elements = command_txt[beg+len(param_sep):end].split( sep = param_sep )
        param = [ read_param(el) for el in elements ]

        command_txt = command_txt[ end+1: ]

    elif (end <= 0)^(beg < 0) :
        raise Exception( f"Parameter brackets not closed in line:\n\t{line}" )
    

    modifier_expr = command_txt.strip()

    return axis, sname.strip(), param, dict(modif), modifier_expr

# Replaces all variable instance with their values
# and evaluates final value of expression
def read_param( expression:str ):
    global variables
    
    try:
        out = eval(expression, variables)
    except:
        raise Exception( 
            "Unrecognized variable or expression found in file, could not match "+ 
            f"\"{expression}\" with given list of variables': \n"+
            ''.join([" -" + el + "\n" for el in variables.keys()]) 
        )
    
    return out

# Based on coded axis calculates eventual rotation 
# scaling factors if given.
def read_axis( axis:str ):
    global channels
    global field_scaling
    global angle_scaling

    ax = axis.strip()

    ax_sep = ax.find( ":" )

    if ax_sep == -1:
        ch = [channels[axis]]
        scale = [1.]
    else:
        ang_sep = ax.find( "/" )

        ax1 = ax[:ax_sep].strip()
        ax2 = ax[ax_sep+1:ang_sep].strip()

        ch = [channels[ax1], channels[ax2]]

        angle = read_param( ax[ang_sep+1:] )*angle_scaling
        scale = [ np.cos(angle), np.sin(angle)*field_scaling[ax1]/field_scaling[ax2] ]
            
    return ch, scale

# Reads proper instructions using "decompose_command" function
# and transforms them into "impulse" elements, which are then 
# gathered in single list. This is functions output
def gather_impulses( instructions ):
    global variables

    impulses = []
    t_current = 0.0
    total_dur = 0.0

    for command in instructions:

        variables["get_current_time()"] = t_current

        axis, gname, param, modif, modif_expr = decompose_command( command )

        new_shape = shapes_dict[ gname ]

        if "T" in modif.keys():
            duration = modif["T"]
        else:
            duration = new_shape.get_duration( param )

        if "Tstart" in modif.keys():
            t_start = modif["Tstart"]
            t_current = t_start + duration
        else:
            t_start = t_current
            t_current += duration

        chan, rotation = read_axis( axis )

        for ch, rot_scaling in zip( chan, rotation ):
            impulses.append( 
                Impulse( 
                    t_start, 
                    duration, 
                    ch, 
                    gname, 
                    param, 
                    scale = rot_scaling,
                    expr = modif_expr
                ) 
            )


        total_dur = max( t_current, t_start + duration, total_dur )

    return impulses, total_dur

# Loops over impulses gathered in "impulses" list and evaluates 
# them for a given times and channel. Only one channel at a time.
def calculate_final_voltages( impulses, t, ch=0 ) -> np.ndarray:
    V = np.zeros_like( t )

    for impulse in impulses:
        V += impulse(t, ch)

    return V

# Goes over list of impulses that are for a channel and tries
# to evaluate when gate logic should be toggled. Outputs 
# boundary time values.
def get_gate_schedule( impulses, ch=0 ):
    
    t_all = []
    add = []

    for impulse in impulses:
        if impulse.ch == ch:
            t_all.append( impulse.t_i )
            t_all.append( impulse.t_f )
            add.append( 1 )
            add.append( -1 )

    i_sort = sorted(range(len(t_all)), key=lambda x:t_all[x])

    t_all = [ t_all[i] for i in i_sort ]
    add = [ add[i] for i in i_sort ] 

    t_fin = []
    u = 0

    for t,du in zip( t_all, add ):
        if (u == 1) and (du == -1):
            t_fin.append(t)
        elif (u == 0) and (du == 1):
            t_fin.append(t)

        u += du

    i = 0
    while i < len(t_fin)-1:
        if t_fin[i] == t_fin[i+1]:
            t_fin.remove( t_fin[i] )
            t_fin.remove( t_fin[i] )
            i -= 1
        else:
            i += 1

    return np.array( t_fin )

# Visualises signal with additional names of shapes.
def draw_instructions( impulses:list[Impulse], signals ):

    Vmax = 0.
    colors = []

    tmin = np.inf
    tmax = 0.0

    for t,V in signals:
        Vmax = max(Vmax, np.max(np.abs(V)))
        tmin = min( tmin, np.min(t) )
        tmax = max( tmax, np.max(t) )

        plot, = plt.plot( t, V, '.-', zorder=10 )
        colors.append( plot.get_color() )

        ylims = np.array([-1,1]) * Vmax * 1.1
        plt.ylim( ylims )
        plt.grid()

    plt.legend( channels.keys() )

    seq = []
    stack = []

    duration = tmax - tmin
    near = 0.05*duration

    for impulse in impulses:

        label = impulse.sname + ("\n" + ", ".join( [f"{p:.2f}" for p in impulse.p] ) if not( impulse.p is None) else "")

        x_pos = impulse.t_i + impulse.T/2
        dist = np.abs(np.array(seq)-x_pos)

        if len(seq) == 0 or np.min( dist ) > near:
            seq.append( x_pos )
            stack.append(0)
            i = len(seq)-1
        else:
            i = np.where( dist <= np.min(dist) )[0][0]
            stack[i] += 1

        y_pos = -Vmax / 3 * ( 1 + stack[i]*0.5 )

        plt.text( 
            x = x_pos, y = y_pos, s = label, color = colors[ impulse.ch ],
            bbox={'facecolor':'white','alpha':0.4,'edgecolor':'black','pad':4},
            ha='center', va='center', zorder = 20
        )

    plt.show()
