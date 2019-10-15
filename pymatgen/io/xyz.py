# coding: utf-8
# Copyright (c) Pymatgen Development Team.
# Distributed under the terms of the MIT License.

import re
import numpy as np

from typing import Dict, List, Tuple, Union, Optional, Hashable, Any
from monty.io import zopen
from pymatgen import IMolecule, IStructure, Molecule, Structure, Lattice
from collections import OrderedDict, namedtuple
from operator import add



"""
Module implementing an (extended) XYZ file object class.
"""

__author__ = "Shyue Ping Ong"
__copyright__ = "Copyright 2012, The Materials Project"
__version__ = "0.1"
__maintainer__ = "Shyue Ping Ong"
__email__ = "shyuep@gmail.com"
__date__ = "Apr 17, 2012"


class XYZ:
    """
    Basic class for importing and exporting Molecules or Structures in XYZ
    format.

    Args:
        mol: Input molecule or list of molecules

    .. note::
        Exporting periodic structures in the XYZ format will lose information
        about the periodicity. Essentially, only cartesian coordinates are
        written in this format and no information is retained about the
        lattice.
    """
    def __init__(self, mol, coord_precision=6):
        if isinstance(mol, Molecule) or not isinstance(mol, list):
            self._mols = [mol]
        else:
            self._mols = mol
        self.precision = coord_precision

    @property
    def molecule(self):
        """
        Returns molecule associated with this XYZ. In case multiple frame
        XYZ, returns the last frame.
        """
        return self._mols[-1]

    @property
    def all_molecules(self):
        """
        Returns all the frames of molecule associated with this XYZ.
        """
        return self._mols

    @staticmethod
    def _from_frame_string(contents):
        """
        Convert a single frame XYZ string to a molecule
        """
        lines = contents.split("\n")
        num_sites = int(lines[0])
        coords = []
        sp = []
        coord_patt = re.compile(
            r"(\w+)\s+([0-9\-\+\.eEdD]+)\s+([0-9\-\+\.eEdD]+)\s+([0-9\-\+\.eEdD]+)"
        )
        for i in range(2, 2 + num_sites):
            m = coord_patt.search(lines[i])
            if m:
                sp.append(m.group(1))  # this is 1-indexed
                # this is 0-indexed
                # in case of 0.0D+00 or 0.00d+01 old double precision writing
                # replace d or D by e for ten power exponent
                xyz = [val.lower().replace("d", "e") for val in m.groups()[1:4]]
                coords.append([float(val) for val in xyz])
        return Molecule(sp, coords)

    @classmethod
    def from_string(cls, contents):
        """
        Creates XYZ object from a string.

        Args:
            contents: String representing an XYZ file.

        Returns:
            XYZ object
        """
        if contents[-1] != "\n":
            contents += "\n"
        white_space = r"[ \t\r\f\v]"
        natoms_line = white_space + r"*\d+" + white_space + r"*\n"
        comment_line = r"[^\n]*\n"
        coord_lines = r"(\s*\w+\s+[0-9\-\+\.eEdD]+\s+[0-9\-\+\.eEdD]+\s+[0-9\-\+\.eEdD]+\s*\n)+"
        frame_pattern_text = natoms_line + comment_line + coord_lines
        pat = re.compile(frame_pattern_text, re.MULTILINE)
        mols = []
        for xyz_match in pat.finditer(contents):
            xyz_text = xyz_match.group(0)
            mols.append(cls._from_frame_string(xyz_text))
        return XYZ(mols)

    @classmethod
    def from_file(cls, filename):
        """
        Creates XYZ object from a file.

        Args:
            filename: XYZ filename

        Returns:
            XYZ object
        """
        with zopen(filename) as f:
            return cls.from_string(f.read())

    def _frame_str(self, frame_mol):
        output = [str(len(frame_mol)), frame_mol.composition.formula]
        fmtstr = "{{}} {{:.{0}f}} {{:.{0}f}} {{:.{0}f}}".format(self.precision)
        for site in frame_mol:
            output.append(fmtstr.format(site.specie, site.x, site.y, site.z))
        return "\n".join(output)

    def __str__(self):
        return "\n".join([self._frame_str(mol) for mol in self._mols])

    def write_file(self, filename):
        """
        Writes XYZ to file.

        Args:
            filename: File name of output file.
        """
        with zopen(filename, "wt") as f:
            f.write(self.__str__())

class EXYZ(XYZ):
    """
    Basic class for importing and exporting structure or molecules in the extended XYZ
    format as described at https://libatoms.github.io/QUIP/io.html#extendedxyz.

    Args:
        structure: Input (list of) structure(s) or molecule(s)

    .. note::
        While exporting periodic structures in the XYZ format will lose information
        about periodicity, the extended XYZ format does retain such information.
        Moreover, arbitrary metadata is retained and encoded in terms of bools
        (T or F), integer numbers, floats or strings (delimited by quotation marks
        when including whitespaces) on a per-site and per-structure/molecule basis.
    """

    _code2type = {
        "L" : bool, "l" : bool,
        "I" : int, "i" : int,
        "R" : float, "r" : float,
        "S" : str, "s" : str
    }
    _type2code = {
        bool : "L",
        int : "I", np.int : "I", np.int64 : "I", np.int32 : "I", np.int16 : "I", np.int8 : "I",
        float : "R", np.float : "R", np.float128 : "R", np.float64 : "R", np.float16 : "R",
        str : "S", list : "S", tuple : "S", np.ndarray : "S"
    }

    _quotes = r"\"'`´"
    _whites_etc = r"\t\n\r\f\v"
    _delims = r"=:"

    _site_prop_key_sanitize_match = "[" + _whites_etc + _quotes + ":" + "]+"
    _site_prop_val_sanitize_match = "[" + _whites_etc + _quotes + "]+"
    _frame_prop_key_sanitize_match = "[" + " " + _whites_etc + _quotes + "=" + "]+"
    _frame_prop_val_sanitize_match = "[" + _whites_etc + _quotes + "=" + "]+"

    EXYZData = namedtuple("EXYZData", ["code", "data", "widths"])

    def __init__(
            self,
            mol: Union[IStructure, List[IStructure], IMolecule, List[IMolecule]],
            mol_props: Union[Dict, List[Dict], Tuple[Dict]] = None,
            float_precision : int = 6
    ) -> None:
        super().__init__(
            mol,
            coord_precision=float_precision
        )

        self._fmt_float = "{{:.{}f}}".format(self.precision)

        if isinstance(mol_props, (List, Tuple)):
            self._mols_props = mol_props
        elif mol_props is not None:
            self._mols_props = [mol_props]
        else:
            self._mols_props = [None for m in self._mols]

        if not len(self._mols_props) == len(self._mols):
            raise ValueError(
                "not as many molecule property sets ({}) as molecules ({})".format(
                    len(self._mols_props),
                    len(self._mols)
                )
            )

    def _site_prop_key(
            self,
            k: str
    ) -> str:
        key = re.sub(self._site_prop_key_sanitize_match, "", str(k))
        return key

    def _site_prop_val(
            self,
            v: str
    ) -> str:
        val = re.sub(self._site_prop_val_sanitize_match, "", str(v))
        if re.search("[ ]+", val):
            val = '"' + val + '"'
        return val

    def _code(
            self,
            val: Any
    ) -> str:
        try:
            code = self._type2code[type(val)]
        except KeyError:
            raise ValueError(
                "Unable to map {} ({}), use appropriate string representation".format(type(val), val)
            )
        return code

    def _val2coldata(
            self,
            val: Union[List, Tuple, np.ndarray, Any],
            probe_seq=True
    ) -> (str, List[str], List[int]):
        if probe_seq and isinstance(val, (list, tuple, np.ndarray)):
            codes = [self._code(v) for v in val]
            if not len(set(codes)) == 1:
                raise TypeError("Inconcistent types in data field")
            code = codes[0]
            data_str = [self._site_prop_val(*self._val2coldata(v, probe_seq=False)[1]) for v in val]
            widths = [len(d) for d in data_str]
        else:
            code = self._code(val)
            if code == "R":
                data_str = self._fmt_float.format(val)
            elif code == "L":
                data_str = "T" if val else "F"
            else:
                data_str = str(val)
            data_str = [self._site_prop_val(data_str)]
            widths = [len(data_str[0])]
        return code, data_str, widths

    def _site_data_columns(
            self,
            data_col: List
    ) -> EXYZData:
        code0 = None
        data_str0 = []
        widths0 = None
        for d in data_col:
            code, data_str, widths = self._val2coldata(d)
            if not code0:
                code0 = code
            if not widths0:
                widths0 = widths
            if not code0 == code:
                raise TypeError("Inconsistent types in data column")
            if not len(widths0) == len(widths):
                raise ValueError("Inconsistent lengths in data column")
            data_str0.append(data_str)
            widths = tuple(max(w, w0) for w, w0 in zip(widths, widths0))
            widths0 = widths
        return EXYZ.EXYZData(code=code, data=data_str0, widths=widths)


    def _site_prop_keys_and_data(
            self,
            mol: IStructure,
            data: OrderedDict
    ) -> str:
        props_str = "species"
        data["species"] = self._site_data_columns([str(site.specie) for site in mol])
        props_str += ":" + data["species"].code + ":" + str(len(data["species"].widths))

        props_str += ":" + "pos"
        data["pos"] = self._site_data_columns([site.coords for site in mol])
        props_str += ":" + data["pos"].code + ":" + str(len(data["pos"].widths))

        for (key, val) in mol.site_properties.items():
            key = self._site_prop_key(str(key))
            props_str += ":" + key
            data[key] = self._site_data_columns(val)
            props_str += ":" + data[key].code + ":" + str(len(data[key].widths))

        # delimit properties value with quotes in case property keys include space(s):
        if re.match("[ ]+", props_str):
            props_str = '"' + props_str + '"'
        return "Properties=" + props_str

    def _frame_prop_key(
            self,
            key: str
    ) -> str:
        key = re.sub(self._frame_prop_key_sanitize_match, "", str(key))
        return key

    def _frame_prop_val(
            self,
            val: str
    ) -> str:
        val = re.sub(self._frame_prop_val_sanitize_match, "", str(val))
        if re.search("[ ]+", val):
            val = '"' + val + '"'
        return val

    def _val2propval(
            self,
            val: Any
    ) -> str:
        code = self._code(val)
        if code == "R":
            val = self._fmt_float.format(val)
        elif code == "L":
            val = "T" if val else "F"
        else:
            val = self._frame_prop_val(str(val))

        return val

    def _get_commentline_and_data(
            self,
            mol: IStructure,
            data: OrderedDict,
            props: Dict
    ) -> str:
        com_str = 'Lattice="' + " ".join(
            self._fmt_float.format(x) for x in mol.lattice.matrix.flat
        ) + '"'

        com_str += " " + self._site_prop_keys_and_data(mol, data)

        if props:
            for (key, val) in props.items():
                if not isinstance(key, str):
                    raise TypeError("non-string frame property key")
                key = self._frame_prop_key(key)
                com_str += " " + key
                com_str += "=" + self._val2propval(val)
        return com_str

    def _compile_data_lines(
            self,
            data: OrderedDict
    ) -> List[str]:
        if not len(set(len(col.data) for col in data.values())) == 1:
            raise ValueError("inconsistent amount of properties given.")

        prop, col = data.popitem(last=False)
        lines = [" ".join(f.rjust(w) for (f,w) in zip(fields, col.widths)) for fields in col.data]

        for col in data.values():
            lines = map(
                add,
                lines,
                [" " + " ".join(f.rjust(w) for (f,w) in zip(fields,col.widths)) for fields in col.data]
            )

        return lines

    def _frame_str(
            self,
            mol: Union[IMolecule, IStructure],
            props: Dict = None
    ) -> str:
        if not mol.lattice:
            lat = Lattice.cubic(2.*mol.distance_matrix.max())
            center = lat.get_cartesian_coords([.5,.5,.5])
            mol = IStructure(
                lat,
                [s.specie for s in mol],
                [s.coords - mol.center_of_mass + center for s in mol],
                coords_are_cartesian=True,
                site_properties=mol.site_properties
            )

        output = [str(mol.num_sites)]

        data = OrderedDict()
        output.append(self._get_commentline_and_data(mol, data, props))

        output.extend(self._compile_data_lines(data))

        return "\n".join(output)

    def __str__(self) -> str:
        return "\n".join(self._frame_str(m, p) for m, p in zip(self._mols, self._mols_props))