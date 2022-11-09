meta_info = """
// summary: G-code for Synthetic Vascular Printing
// authors : Zachary Sexton & Jessica Herrmann
// email  : zsexton@stanford.edu
"""

licence = """
// Copyright (c) Stanford University, The Regents of the University of
//               California, and others.
//
// All Rights Reserved.
//
// See Copyright-SimVascular.txt for additional details.
//
// Permission is hereby granted, free of charge, to any person obtaining
// a copy of this software and associated documentation files (the
// "Software"), to deal in the Software without restriction, including
// without limitation the rights to use, copy, modify, merge, publish,
// distribute, sublicense, and/or sell copies of the Software, and to
// permit persons to whom the Software is furnished to do so, subject
// to the following conditions:
//
// The above copyright notice and this permission notice shall be included
// in all copies or substantial portions of the Software.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
// IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
// TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
// PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER
// OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


"""

variables = """
// --------- Variable Declarations ----------- //
//    Variables                   Units            Description
DVAR $syringe_diameter            mm               diameter of syringe body
DVAR $nozzle_diameter             mm               diameter of nozzle outlet
DVAR $jog_speed                   mm/sec           movement speed while not printing
DVAR $overall_length              mm
DVAR $print_speed                 mm/sec           movement speed while printing
DVAR $extrusion_coeff             None             adjustment ratio between print speed and nozzle size
DVAR $plunger_rate                mm/sec           rate of syringe plunger depression
DVAR $x_start                                      starting x value
DVAR $y_start                                      starting y value
DVAR $z_start                                      starting z value
DVAR $x_offset                    mm               reference frame difference x direction
DVAR $y_offset                    mm               reference frame difference y direction
DVAR $z_offset                    mm               reference frame difference z direction
DVAR $initial_pump_distance       mm               starting distance for pump plunger
DVAR $initial_pump_rate           mm/sec           pump plunger rate

"""

initialize_variables = """
// --------- General Parameters ----------- //
$syringe_diameter      = 9.6        // 3BB Syringe = 9.55 mm, 10mL glass = 10.3mm
$jog_speed             = 10
$initial_pump_distance = 0.05
$initial_pump_rate     = 5

// --------- Tube Parameters ----------- //
$nozzle_diameter = 0.84
$print_speed     = 6
$extrusion_coeff = 1

// --------- Calculated Parameters ----------- //
$plunger_rate = $extrusion_coeff * $print_speed * $nozzle_diameter**2/$syringe_diameter**2

"""

stop_pump = """
Call StopExtrude\n
"""

start_pump = """
Call StartExtrude P$initial_pump_distance Q$initial_pump_rate R$plunger_rate
G90 F3
G108

"""

recalculate_plunger_rate = """
$extrusion_Boeff = {}
$plunger_rate = $extrusion_coeff * $print_speed * $nozzle_diameter**2 / $syringe_diameter**2

"""

branch_name = """
// Branch {}

"""

end_file = """
G90
G1 B60

M2

"""

start_extrude = """
DFS StartExtrude

	Enable Aa
	G91
	G1 Aa-$initial_pump_dist E$initial_pump_rate
	M104
	S$plunger_rate

EndDFS

"""

stop_extrude = """
DFS StopExtrude

	M5
	Enable Aa
	G91
	G1 Aa$initial_pump_dist E$initial_pump_rate

EndDFS
"""

def g_configure(x,y,z,precision=3):
    return "G90\nG92 X{} Y{} B{}\n\n".format(round(x,precision),round(y,precision),round(z,precision))

def g_coordinate(x,y,z,precision=3):
    return "G1 X{} Y{} B{}\n".format(round(x,precision),round(y,precision),round(z,precision))

def generate_gcode(paths,offset=[0,0,0]):
    file = open("test.gcode","w+")
    filetext = licence + meta_info + variables + initialize_variables
    for idx,path in enumerate(paths):
        filetext += branch_name.format(idx)
        filetext += g_configure(path[0][0],path[0][1],path[0][2])
        filetext += stop_pump
        filetext += start_pump
        for jdx,point in enumerate(path):
            filetext += g_coordinate(point[0],point[1],point[2])
        filetext += stop_pump
    filetext += end_file
    filetext += start_extrude
    filetext += stop_extrude
    file.write(filetext)
