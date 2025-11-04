import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline, interp1d


axis_impulse_sep = "<-"
variable_ovewrite_chr = ":="

modif_sep = ","
modif_set_chr = "="
param_sep = ","
signal_variable = "signal_shape_base_value"

axis_rot_sep = ":"
axis_angle_sep = "/"

modif_lbracket = "["
modif_rbracket = "]"

param_lbracket = "("
param_rbracket = ")"

impulse_duration_mod = "T"
impulse_start_time_mod = "Tstart"

class Shape:
    """
    Class for storing signal shape as a set of possible shapes 
    of curve. Also used as general curve shape for experiment 
    setup like guiding field shape etc. 

    It functionality is to store information about:
     - a shape function V for x in [0,1]
     - duration of shape, as constant T or function Tf
       with the same parameters as shape function
    """
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
	"sin": Shape(    lambda x, A=1.0, phi=0.0, N_cycl=1.0: A*np.sin( N_cycl * x * 2 * np.pi + phi*np.pi ),
),
	"line": Shape(
    lambda x, V0=0.0, V1=0.0: V0+(V1-V0)*x,
),
	"poly": Shape(
    lambda x, *par: np.polyval( np.flip(par), x )
),
	"exp": Shape(
    lambda x, dx=1.0, V=0.0, dV=0.0: V + dV * ( 1 - np.exp(x/dx) ) / ( 1 - np.exp(1/dx) ),
)
}

# dictionary for numbering axis
channels:dict[str,int] = {}
# dictionary with custom variables
variables:dict[str,float] = {}
# dictionary with coil calibration
field_scaling:dict[str,float] = {}

# scaling factor to use proper angle unit.
# 1 for radians, pi/180 for degrees
angle_scaling = np.pi/180

class Impulse:
    """    
    Class that is a primary element of signal. It represents 
    a single channel impulse with defined starting and ending 
    time.     
    When called it will give 0 when outside of its 
    duration or not proper channel. In other case it will 
    interpolate given shape stretched for whole duration.
    """
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

        temp_dict = {}
        temp_dict.update(variables)

        def V_singular( t, ch=0 ):
            if (t > self.t_i) and (t <= self.t_f) and (ch == self.ch):
                return self.e( self.g.get_curve( ( t - t_start )/duration, param ) )*self.s
            else:
                return 0.0

        self.V = np.vectorize( V_singular, excluded=[1] )

    def __call__( self, t, ch ):
        return self.V(t,ch)

def load_parameters( new_variables = None, new_shapes = None, new_channels = None, new_field_scaling=None ):
    """function to set global parameters, since they don't easily transfer file to file"""
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

def shape_from_file_cubic( fname: str, boundary_condition: str = "natural" ) -> Shape:
    """
    Allows to read file with two columns, time and field/voltage, to
    create Shape element representing it. It uses CubicSpline function 
    from Scipy.Interpolate module to obtain interpolation spline of 
    data from file. 
    
    By default boundary condition is set to "natural", meaning at the
    edges second derivative is set to 0. It is directly forwarded
    to CubicSpline, so go there to learn more bc options.

    In case file with data is in other directory also add path
    in fname. The safest is to write full path.
    """
    data = np.loadtxt( fname )

    time = data[:,0]
    signal = data[:,1]

    time = time - time[0]
    duration = time[-1]

    func = CubicSpline( time/duration, signal, bc_type=boundary_condition )

    return Shape( curve = func, duration = duration )

def shape_from_file( fname: str, interp_type: str = "linear" ) -> Shape:
    """
    Allows to read file with two columns, time and field/voltage, to
    create Shape element representing it. It uses interp1d function 
    from Scipy.Interpolate module to obtain interpolation of data 
    from file. 
    
    By default interpolation type is set to "linear", meaning at the
    points are gonna be connected by straight lines. It is directly 
    forwarded to interp1d, so go there to learn more interpolation
    options.

    In case file with data is in other directory also add path
    in fname. The safest is to write full path.
    """
    data = np.loadtxt( fname )

    time = data[:,0]
    signal = data[:,1]

    time = time - time[0]
    duration = time[-1]

    func = interp1d( time/duration, signal, kind = interp_type )

    return Shape( curve = func, duration = duration )

def get_commands_from_instructions( instructions_txt:str ) -> list[str]:
    """
    Reads string "instructions_txt" with raw instruction lines
    and cleans it from comments (text after "#" character) and
    omits empty lines
    """

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

def decompose_shape_command( line:str ) -> tuple[str,str,list[str],dict[str,float],str] :
    """
    decompose_shape_command(line) splits command
    given as string "line" variable into four parts:
    - shape name
    - general modifiers from square brackets
    - shape specific parameters from normal brackets
    - axis indicator from right side of ">" symbol
    It does convert variables to values.
    """

    m = line.find( axis_impulse_sep )

    command_txt = line[m+len(axis_impulse_sep):]
    axis = line[:m].strip()

    # find square
    modif = []

    beg = command_txt.find(modif_lbracket)
    end = command_txt.find(modif_rbracket)
    if ( beg >= 0 ) and ( end > 0 ):

        elements = command_txt[beg+len(modif_sep):end].split( sep = modif_sep )
        elements = [ el.split( sep = modif_set_chr ) for el in elements ]
        modif = [ ( el[0].strip(), read_param(el[1]) ) for el in elements ]

        # cutoff modifiers from command text
        command_txt = command_txt[end+1:]

    elif (end <= 0)^(beg < 0) :
        raise Exception( f"Modifier brackets not closed in line:\n\t{line}" )

    # Names sorted in decreasing length order in case 
    # some name is inside other, for example:
    #    x <- strange_sin(2)
    # if we would look if "sin" exists first 
    # we would've incorrectly identify shape
    def_shape_names = sorted(shapes_dict.keys(), key=lambda x: len(x), reverse=True)

    cmd_shape = ""
    for name in def_shape_names:
        shape_loc = command_txt.find(name)

        if shape_loc >= 0:
            cmd_shape = name
            break

    # in case no shape was given program assumes 
    # it to be constant signal of value given.
    if shape_loc < 0:
        return axis, "line", [ read_param( command_txt ) ], dict(modif), lambda x: x


    # checking if there are parameter brackets for shape
    beg = command_txt.find(param_lbracket, shape_loc)

    end = -1

    opened_br = 1

    for curr in range(beg+1, len(command_txt)):

        if command_txt[curr] == param_lbracket:
            opened_br += 1
        if command_txt[curr] == param_rbracket:
            opened_br -= 1

        if opened_br == 0:
            end = curr
            break

    param = None

    if ( end > 0 ) and ( beg >= 0 ):
        parameters_txt = command_txt[beg+len(param_sep):end].strip()
        
        if parameters_txt != "":
            elements = parameters_txt.split( sep = param_sep )
            param = [ read_param(el) for el in elements ]
    
    elif ( end <= 0 ) and ( beg < 0 ):
        end = shape_loc + len(cmd_shape) - 1
        
    else:
        raise Exception( f"Parameter brackets not opened/closed in line:\n\t{line}" )

    expr = command_txt[ :shape_loc ] + signal_variable + command_txt[ end+1: ]

    modifier_expr = read_param( "lambda " + signal_variable + ": " + expr )

    return axis, cmd_shape, param, dict(modif), modifier_expr

def overwrite_variable( line:str ):
    """changes adequate variable in dictionary based on expression"""
    global variables

    m = line.find( variable_ovewrite_chr )

    var_name = line[:m].strip()
    expr = line[m+len(variable_ovewrite_chr):]

    variables.update( {var_name:read_param(expr)} )

    pass

def read_param( expression:str ):
    """Evaluates expression with predefined variables using eval function. 
    Used where real number is expected."""
    global variables

    temp_dict = {}
    temp_dict.update(variables)

    try:
        out = eval(expression, temp_dict)
    except:
        raise Exception(
            "Unrecognized variable or expression found in file, could not match "+
            f"\"{expression}\" with given list of variables': \n"+
            ''.join([" -" + el + "\n" for el in variables.keys()])
        )

    return out

def read_axis( axis:str ) -> tuple[list[str],list[float]]:
    """Based on coded axis calculates eventual rotation
    scaling factors if given."""
    global channels
    global field_scaling
    global angle_scaling

    ax = axis.strip()

    ax_sep_pos = ax.find( axis_rot_sep )

    if ax_sep_pos == -1:
        ch = [channels[axis]]
        scale = [1.]
    else:
        ang_sep = ax.find( axis_angle_sep )

        ax1 = ax[:ax_sep_pos].strip()
        ax2 = ax[ax_sep_pos+1:ang_sep].strip()

        ch = [channels[ax1], channels[ax2]]

        angle = read_param( ax[ang_sep+1:] )*angle_scaling
        scale = [ np.cos(angle), np.sin(angle)*field_scaling[ax1]/field_scaling[ax2] ]

    return ch, scale

def gather_impulses( instructions:list[str] ) -> tuple[list[Impulse], float]:
    """ Main pseudocode interpreter
    Reads proper instructions using "decompose_shape_command" 
    function and transforms them into "Impulse" elements, which 
    are then gathered in single list and outputted. Additionally 
    on output there is total signal duration.
    """
    global variables

    impulses = []
    t_current = 0.0
    total_dur = 0.0

    for command in instructions:

        variables.update( { "current_time": lambda:t_current,
                            "end_time": lambda:total_dur } )

        if variable_ovewrite_chr in command:
            overwrite_variable( command )

        if axis_impulse_sep in command:

            axis, gname, param, modif, modif_expr = decompose_shape_command( command )

            new_shape = shapes_dict[ gname ]

            t_start = t_current
            duration = new_shape.get_duration( param )

            if impulse_duration_mod in modif.keys():
                duration = modif[impulse_duration_mod]

            if impulse_start_time_mod in modif.keys():
                t_start = modif[impulse_start_time_mod]

            t_current = t_start + duration

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

def calculate_final_voltages( impulses, t, ch=0 ) -> np.ndarray:
    """
    Loops over impulses gathered in "impulses" list and evaluates
    them for a given time values. 
    
    Only one channel at a time, which is given through ch. Can be 
    given as either number or name of channel (from channels)
    """
    if type(ch) == str:
        ch = channels[ch]

    V = np.zeros_like( t )

    for impulse in impulses:
        V += impulse(t, ch)

    return V

def get_gate_schedule( impulses, ch=0 ):
    """
    Goes over list of impulses that are for a channel and tries
    to evaluate when gate logic should be toggled. Outputs
    boundary time values.
    """
    
    if type(ch) == str:
        ch = channels[ch]
    
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

def draw_instructions( impulses:list[Impulse], signals ):
    """Visualizes signal with additional names of shapes."""
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

        param_txt = ""
        if not( impulse.p is None ):
            param_txt = "\n" + ", ".join( [f"{p:.2f}" for p in impulse.p] )

        label = impulse.sname + param_txt

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
    
    variables_txt = []
    for key in variables:

        val = variables[key]
        if type(val) is float:
            variables_txt.append( key + f": {val}" )

    variables_txt = "\n".join( variables_txt )
    plt.text(
        x = tmin, y = -Vmax, s = variables_txt,
        bbox={'facecolor':'white','alpha':0.4,'edgecolor':'black','pad':4},
        ha='left', va='bottom', zorder = 20
    )

    plt.show()
