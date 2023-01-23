
eindhoven = {'expname': 'eindhoven',
             'x0'     : 152000,
             'y0'     : 375000,
             'itot'   : 440,
             'jtot'   : 320,
             'dx'     : 50,
             'dy'     : 50,                    
             'nprocx' : 4,
             'nprocy' : 4 
            }

test       = {'expname': 'test',
             'x0'     : 912500,
             'y0'     : 940000,
             'itot'   : 64,
             'jtot'   : 64,
             'dx'     : 400,
             'dy'     : 400,                    
             'nprocx' : 2,
             'nprocy' : 2 
            }

ruisdael   = {'expname': 'ruisdael',
             'x0'     : 910000,
             'y0'     : 940000,
             'itot'   : 864,
             'jtot'   : 576,
             'dx'     : 200,
             'dy'     : 200,                    
             'nprocx' : 12,
             'nprocy' : 12 
            }

smalldomain = {'expname': 'smalldomain',
            'x0' : 155000,
            'y0' : 386000,
            'itot' : 128,
            'jtot' : 128,
            'dx' : 50,
            'dy' : 50,
            'nprocx' : 4,
            'nprocy' : 4
            }

domains = {'eindhoven': eindhoven,
           'eindhoven_small': smalldomain,
           'test'     : test,
           'ruisdael' : ruisdael 
            }
