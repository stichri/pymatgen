#!/usr/bin/env python3

from pymatgen.io.vasp.inputs import Poscar
from pymatgen.io.xyz import EXYZ

import re
import random
import numpy as np
import pymatgen as pmg



strctr = Poscar.from_file("test_files/POSCAR.Fe3O4").structure
v = [np.array([random.uniform(-1000,1000),random.uniform(-100,100),random.uniform(-10,10)]) for s in strctr]
sd = [[True,True,True] for s in strctr]
strctr = strctr.copy(
	site_properties={"velocities":v, "selective_dynamics":sd}
)

dct = {
	"floatval" : 0.0,
	"bool1" : True,
        "bool2" : False,
        "intval" : 1,
        "strval" : "str",
        "strval2" : "str str",
        "strval3" : "str\n=str",
        "Lattice" : True,
        "Foo Bar" : 1,
        "Left=Right" : True,
	"vecfloatval" : [1.,2.,3.]
}
#print(dct)

exyz = EXYZ(
	strctr,
	dct
)
#print(exyz)

exyz = EXYZ(
	pmg.Structure(
		pmg.Lattice.cubic(1.0),
		["V", "V"],
		[np.array([.0,.0,.0]), np.array([.5,.5,.5])],
		site_properties={
			"intvals": [1,2],
			"floatvals": [1.0,2.0],
			"strvals": ["str1", "str2"],
			"vecintvals": [[1,1,1],[0,0,0]],
			"vecstrvals": [[True,True,True],[False,False,False]],
			"vecwhitestrvals": [["a b", "bc", "c d"],["ab", "b c", "cd"]]
		}
	)
)

exyz.write_file("test.exyz")
exyz = EXYZ.from_file("test.exyz")
