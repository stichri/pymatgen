# coding: utf-8
# Copyright (c) Pymatgen Development Team.
# Distributed under the terms of the MIT License.

import re

from typing import Dict, List, Union, Optional, Hashable
from monty.io import zopen
from pymatgen import IMolecule, IStructure, Molecule, Structure, Lattice
from collections import OrderedDict


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

    @staticmethod
    def from_string(contents):
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
            mols.append(XYZ._from_frame_string(xyz_text))
        return XYZ(mols)

    @staticmethod
    def from_file(filename):
        """
        Creates XYZ object from a file.

        Args:
            filename: XYZ filename

        Returns:
            XYZ object
        """
        with zopen(filename) as f:
            return XYZ.from_string(f.read())

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
    Basic class for importing and exporting structures or molecules in extended XYZ
    format as described at https://libatoms.github.io/QUIP/io.html#extendedxyz.

    Args:
        structure: Input structure/molecule or list of structures/molecules

    .. note::
        While exporting periodic structures in the XYZ format will lose information
        about the periodicity, the extended XYZ format does retain such information.
        Moreover, arbitrary metadata is retained and encoded in terms of bools
        (T or F), integer numbers, floats or strings (delimited by quotation marks
        when including whitespaces) on a per-site and per-structure/molecule basis.
    """

    _quotes = "\"'`´"
    _whites_etc = r"\t\n\r\f\v"
    _delims = "=:"
    _frameval_sanitize_regex = r"[" + _quotes + _whites_etc + "=" + r"]+"
    _framekey_sanitize_regex = r"[" + _quotes + _whites_etc + "=" + " " + r"]+"
    _colkey_sanitize_regex = r"[" + _whites_etc + " " + _delims + r"]+"

    _type_lookup = {
        "L" : bool, "l" : bool,
        "I" : int, "i" : int,
        "R" : float, "r" : float,
        "S" : str, "s" : str
    }
    _code_lookup = {
        bool : "L",
        int : "I",
        float : "R",
        str : "S"
    }

    def __init__(
            self,
            mol: Union[IStructure, List[IStructure], IMolecule, List[IMolecule]],
            mol_props: Optional[Union[Dict, List[Dict]]] = None,
            coord_precision: int = 6
    ) -> None:
        super().__init__(
            mol,
            coord_precision=coord_precision
        )

        self._fmt_float = "{{:.{}f}}".format(self.precision)

        if isinstance(mol_props, list):
            self._mols_props = mol_props
        elif mol_props is not None:
            self._mols_props = [mol_props]
        else:
            self._mols_props = [None for m in self._mols]

        if len(self._mols_props) != len(self._mols):
            raise ValueError(
                "not as many dicts ({}) as structures/molecules ({})".format(
                    len(self._mols_props), len(self._mols)
                )
            )

    def _lattice_to_prop(
            self,
            lat: Lattice
    ) -> str:
        return '"' + ' '.join(self._fmt_float.format(x) for x in lat.matrix.flat) + '"'

    def _val_to_frameval(
            self,
            val: Union[bool, int, float, str]
    ) -> str:
        if isinstance(val, bool):
            str_result = "T" if val else "F"
        elif isinstance(val, int):
            str_result = str(val)
        elif isinstance(val, float):
            str_result = self._fmt_float.format(val)
        else:
            str_result = re.sub(self._frameval_sanitize_regex, "", str(val))
            if re.search(r"[ ]+", str_result) is not None:
                str_result = '"' + str_result + '"'
        return str_result

    def _val_to_framekey(
            self,
            val: Union[bool, int, float, str]
    ) -> str:
        frameprop = re.sub(self._framekey_sanitize_regex, "", str(val))
        return frameprop

    def _val_to_colkey(
            self,
            val: Hashable
    ) -> str:
        return re.sub(self._colkey_sanitize_regex, "", val)

    def _get_properties_tag(
            self,
            props: Dict = None
    ) -> str:
        props_tag: str = "species:S:1:pos:R:3"

        if props:
            for (k, v) in props.items():
                props_tag += ":" + self._val_to_colkey(k)

        return props_tag

    def _frame_str(
            self,
            mol: Union[IMolecule, IStructure],
            props: Dict = None
    ) -> str:
        output = [str(mol.num_sites)]

        prop_line_dict = OrderedDict()
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
        prop_line_dict["Lattice"] = self._lattice_to_prop(mol.lattice)
        prop_line_dict["Properties"] = self._get_properties_tag(props)

        if mol.site_properties:
            for (key, val) in mol.site_properties.items():
                if key not in prop_line_dict.keys():
                    prop_line_dict[self._val_to_framekey(key)] = self._val_to_frameval(val)
        output.append(
            " ".join("{}={}".format(k, v) for (k, v) in prop_line_dict.items())
        )

        return "\n".join(output)

    def __str__(self):
        return "\n".join(
            [self._frame_str(m,p) for (m,p) in zip(self._mols, self._mols_props)]
        )