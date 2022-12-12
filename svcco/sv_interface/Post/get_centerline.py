#!/usr/bin/env python

import os
import sys
import pdb
import tempfile
import shutil

from get_database import Database, input_args

# sys.path.append('/home/pfaller/work/repos/SimVascular_fork/Python/site-packages/')
sys.path.append('/home/pfaller/work/repos/SimVascular_zasexton/Python/site-packages/')

from sv_rom_simulation import centerlines


# list of geometries whos mesh is so coarse the vmtk centerline extraction failes
geos_coarse = ['0119_0001', '0085_1001', '0084_0001']
# completed: '0087_1001', '0005_1001',
# no fine surface mesh: '0082_1001'

def add_suffix(fpath, suffix):
    ext = os.path.splitext(fpath)
    return ext[0] + suffix + ext[1]


class Params(object):
    """
    Minimal parameter set for Centerlines class
    """
    def __init__(self, p):
        suffix = '_resampled'
        self.boundary_surfaces_dir = p['f_surf_caps']
        self.inlet_face_input_file = p['f_inflow']
        self.surface_model = p['f_surf_in']
        self.output_directory = os.path.dirname(add_suffix(p['f_outlet'], suffix))
        self.CENTERLINES_OUTLET_FILE_NAME = os.path.basename(add_suffix(p['f_outlet'], suffix))
        self.centerlines_output_file = add_suffix(p['f_cent_out'], suffix)
        self.cent_out = add_suffix(p['f_cent_out'], suffix)


def main(db, geometries):
    for geo in geometries:
        if not db.get_surfaces(geo, 'all_exterior'):
            continue

        if geo in geos_coarse:
            f_surf_in = os.path.join('/home/pfaller/work/osmsc/data_generated/mesh_fine', geo + '_fine.vtp')
        else:
            f_surf_in = db.get_surfaces(geo, 'all_exterior')

        # get model paths
        params = {'f_surf_in': f_surf_in,
                  'f_surf_caps': tempfile.mkdtemp(),
                  'f_inflow': os.path.basename(db.get_surfaces(geo, 'inflow')),
                  'f_outlet': db.get_centerline_outlet_path(geo),
                  'f_cent_out_vmtk': db.get_centerline_vmtk_path(geo),
                  'f_cent_out': db.get_centerline_path(geo),
                  'f_surf_out': db.get_surfaces_grouped_path(geo),
                  'f_sections_out': db.get_section_path(geo)}

        # if os.path.exists(params['f_cent_out']):
        #     continue
        print('Running geometry ' + geo)

        # copy cap surfaces to temp folder
        for f in db.get_surfaces(geo, 'caps'):
            shutil.copy2(f, params['f_surf_caps'])

        params = Params(params)

        # call SimVascular centerline extraction
        # try:
        cl = centerlines.Centerlines()
        cl.extract_center_lines(params)
        cl.write_outlet_face_names(params)
        # except Exception as e:
        #     print(e)


if __name__ == '__main__':
    descr = 'Generate a new surface mesh'
    d, g, _ = input_args(descr)
    main(d, g)
