paths="""path_{{}} = pathplanning.Path()
path_{{}}.set_control_points({{}})
if {}:
    dmg.add_path('vessel_{{}}',path_{{}})
"""
