import numpy as np

def generate_input(viscosity, delta_T, x0, y0, radius, fname):
  input_string = """#########################################################
# Reproducing Liu and Gurnis (2008) experiment from paper's section 3.1
# 2-d shell with a single layer having an initial condition of a hot, spherical anomoly in lower mantle
#########################################################

# =================================================
#               Global parameters 
# =================================================

  set Dimension                              = 2
  set Use years in output instead of seconds = true
  set End time                               = 9e6        # total time is 9 million years
  set Output directory                       = ./output_LG_single_{5}

# =================================================
#                 Model Geometry
# =================================================
  
  subsection Geometry model
    set Model name = spherical shell

    subsection Spherical shell
      set Inner radius  = 3473000
      set Outer radius  = 6301000
      set Opening angle = 90
    end
  end

# =================================================
#            Model Compositional Fields
# =================================================

  subsection Material model
    set Model name = simple

    subsection Simple model
      set Thermal expansion coefficient = 3e-5   # (3x10^-5)       from Liu and Gurnis Table 1
      set Viscosity                     = {0}   # (10^21 Pa s)    from Liu and Gurnis Table 1
      set Reference temperature         = 300    # Kelvin --> Liu and Gurnis have 393 Celcius as \delta T
      set Reference density             = 3300   # kg/m^3
      set Reference specific heat       = 1424   # calculated using Liu and G given for K, density, and assume k=10^-6
      # set Thermal viscosity exponent    = 0.0  # not mentioned in Liu and Gurnis

    end
  end

  subsection Boundary velocity model
    #set Zero velocity boundary indicators       = inner, outer, left, right
    set Tangential velocity boundary indicators = top, bottom, left, right   # Free-slip on all sides (tangential velocity V-perp=0, dV/dn=0)
  end

  subsection Heating model
    set List of model names =  shear heating
  end

  subsection Boundary temperature model
    set Fixed temperature boundary indicators = top, bottom
    set List of model names = spherical constant

    subsection Spherical constant
      set Inner temperature = 3974
      set Outer temperature = 3974  #774
    end
  end

  subsection Initial temperature model
    set Model name = function

    subsection Function
      set Variable names = x,y
      set Function constants = delT ={1}, x0={2}, y0={3}, r_c ={4}, Tm=3974
      set Function expression = (((sqrt((x-x0)*(x-x0)+(y-y0)*(y-y0)) <= r_c)) ? \
                                (Tm + Tm*delT*exp(-((x0-x)*(x0-x)+(y0-y)*(y0-y))/(r_c*r_c))) : \
                                 (Tm))
    end

  end

  subsection Gravity model
    set Model name = radial constant  #ascii data

    #subsection Vertical
    #  set Magnitude = 1e4 # = Ra
    #end
  end

# =================================================
#                 Mesh Refinement
# =================================================

  subsection Mesh refinement
    set Initial global refinement          = 5 #4  #used finer mesh to get better circle anomoly
    set Strategy                           = temperature
    set Time steps between mesh refinement = 0
  end

# =================================================
#                 Postprocessing
# =================================================

  subsection Postprocess
    set List of postprocessors = visualization, velocity statistics, temperature statistics, heat flux statistics, depth average

    subsection Visualization
      set Output format                 = vtu
      set Time between graphical output = .5e6  #.45e6     # output every 500,000 years --> gives 18 outputs
      set Number of grouped files       = 0
      set List of output variables      = material properties
      set Interpolate output            = false     # git fix for resolution issue
    end

   # subsection Depth average
   #   set Time between graphical output = .45e6   
   # end

  end""".format(viscosity, delta_T, x0, y0, radius, fname )

  with open(fname+'.prm', 'w') as f:
    f.write(input_string)

counter = 0
for viscosity in ['1e21', '1.4e21', '1.8e21', '2e21', '2.2e21']:
    for delta_T in ['0.099', '0.098', '0.097', '0.096', '0.095']:
        for r_var in [4222000.0, 4554500.0, 4887000.0, 5219500.0, 5552000.0]:
            for theta in [15.0, 30.0, 45.0, 60.0, 75.0]:
                for radius in ['700000', '600000', '650000', '550000', '500000']:
                    fname = "LG_sConfig_{0:04d}".format(counter)
                    x0 = r_var*np.cos(np.radians(theta))
                    y0 = r_var*np.sin(np.radians(theta))
                    generate_input( viscosity, delta_T, x0, y0, radius, fname )
                    counter += 1

