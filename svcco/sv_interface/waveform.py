# Script for determining the general waveform from Kassab
# Am J Physiol Heart Circ Physiol291: H1074–H1087, 2006
# Time in seconds
# Flow in mL/min
import numpy as np
from scipy.interpolate import splprep, splev, interp2d, interp1d, bisplrep, bisplev, LinearNDInterpolator, NearestNDInterpolator
import matplotlib.pyplot as plt
"""
LAD_4500_time = [0, 0.012707403, 0.022236425, 0.028206212, 0.03536751,
                 0.044919475, 0.054459204, 0.064003522, 0.074737821, 0.083081451,
                 0.090232042, 0.097373455, 0.105698731, 0.11639785, 0.123523968,
                 0.1330224, 0.138957008, 0.143698576, 0.150818576, 0.156751654,
                 0.163868595, 0.174546301, 0.182847104, 0.188775594, 0.199454829,
                 0.210135594, 0.222004809, 0.233877084, 0.24693781, 0.258811614,
                 0.273063851, 0.288499949, 0.303934518, 0.322932911, 0.33956205,
                 0.357371991, 0.372811149, 0.389437228, 0.407245641, 0.425058641,
                 0.441680132, 0.461865447, 0.479683036, 0.499880587, 0.518891216,
                 0.537906434, 0.552163259, 0.5628532, 0.57116471, 0.580663142,
                 0.592532358, 0.60202926, 0.610336182, 0.618644633, 0.624576181,
                 0.634077672, 0.643585281, 0.65309289, 0.660231244, 0.668562638,
                 0.674523248, 0.68167231, 0.688827489, 0.695982668, 0.701952455,
                 0.707920713, 0.713885912, 0.718665718, 0.723447054, 0.728225332,
                 0.730625177, 0.735403454, 0.741370182, 0.747343028, 0.754501267,
                 0.761653387, 0.767620115, 0.774770706, 0.784301258, 0.79025575,
                 0.800959457, 0.811655517, 0.819967028, 0.828278538, 0.834208557,
                 0.840137047, 0.847257047, 0.853188595, 0.860305536, 0.866235555,
                 0.872165575, 0.880470967, 0.887587907, 0.89589177, 0.906575594,
                 0.918447868, 0.929133222, 0.942198536, 0.951698497, 0.96357689,
                 0.974263773, 0.987327558, 0.99682446, 1.006319833, 1.015816735,
                 1.026503618, 1.040752795, 1.053813521, 1.069252679, 1.082313405,
                 1.098941013, 1.108436386, 1.125060936, 1.139311642, 1.153560819,
                 1.17018384, 1.186812978, 1.207012058, 1.22839806, 1.24740563,
                 1.265220161, 1.28065473, 1.293712397, 1.30439622, 1.316266966,
                 1.324575417, 1.338830712, 1.351909792, 1.3626135, 1.370946424,
                 1.376905504, 1.381677663, 1.387638273, 1.393605001, 1.398374101]

LAD_4500_flow = [30.40501077, 31.15657663, 31.96620486, 33.00793503, 34.16534744,
                 35.84369869, 37.05873099, 38.44750789, 39.89405217, 40.81965762,
                 41.57166596, 41.97618509, 42.20681213, 42.32131444, 42.14668489,
                 41.79801575, 41.50770396, 41.04379505, 40.63750602, 40.28927936,
                 39.7671606, 39.07085476, 38.37484391, 37.85287264, 37.21448167,
                 36.63400557, 36.05338197, 35.58858812, 35.12364677, 34.71676778,
                 34.36750868, 33.84435748, 33.26329141, 32.62386801, 32.27431392,
                 31.635038, 31.22771654, 30.76233272, 30.06514194, 29.54169576,
                 28.90256733, 28.20508156, 27.85537999, 27.62121316, 27.4451087,
                 27.44274884, 27.26723434, 27.03424745, 26.74364067, 26.39497153,
                 25.81434794, 25.40776394, 24.94341255, 24.53697604, 24.13083451,
                 23.8979951, 23.89681517, 23.89563524, 24.18432464, 24.64661114,
                 25.34085211, 26.03494558, 26.96069852, 27.88645146, 28.92818163,
                 29.91199693, 30.7799825, 31.76394529, 32.80582295, 33.73187088,
                 34.60029892, 35.52634684, 36.45224727, 37.60980718, 38.65138986,
                 39.46131306, 40.3872135, 41.13922184, 42.00676493, 42.46934642,
                 42.75759334, 42.75626592, 42.46565914, 42.17505236, 41.71099596,
                 41.18902469, 40.78273567, 40.37659413, 39.85447537, 39.39041897,
                 38.92636257, 38.40409632, 37.88197756, 37.30179644, 36.83715008,
                 36.37235622, 35.96562472, 35.67442798, 35.38367371, 35.15053933,
                 34.8017227, 34.45261109, 34.04602708, 33.58152821, 33.1749442,
                 32.82612757, 32.36103873, 31.89609738, 31.48877592, 31.02383457,
                 30.61636562, 30.15186675, 29.62856806, 29.22139409, 28.75630525,
                 28.17509169, 27.82553761, 27.64928565, 27.41497134, 27.12303714,
                 26.65750583, 26.07643976, 25.49566868, 25.03102231, 24.50831359,
                 24.10187707, 23.8684477, 24.09848477, 24.38673169, 24.90693307,
                 25.54325916, 26.23764761, 26.93188858, 27.85778901, 28.43634773]
"""
LAD_4500_time = [0, 0.012707403, 0.022236425, 0.028206212, 0.03536751,
                 0.044919475, 0.054459204, 0.064003522, 0.074737821, 0.083081451,
                 0.090232042, 0.097373455, 0.105698731, 0.11639785, 0.123523968,
                 0.1330224, 0.138957008, 0.143698576, 0.150818576, 0.156751654,
                 0.163868595, 0.174546301, 0.182847104, 0.188775594, 0.199454829,
                 0.210135594, 0.222004809, 0.233877084, 0.24693781, 0.258811614,
                 0.273063851, 0.288499949, 0.303934518, 0.322932911, 0.33956205,
                 0.357371991, 0.372811149, 0.389437228, 0.407245641, 0.425058641,
                 0.441680132, 0.461865447, 0.479683036, 0.499880587, 0.518891216,
                 0.537906434, 0.552163259, 0.5628532, 0.57116471, 0.580663142,
                 0.592532358, 0.60202926, 0.610336182, 0.618644633, 0.624576181,
                 0.634077672, 0.643585281, 0.65309289, 0.660231244, 0.668562638,
                 0.674523248, 0.68167231, 0.688827489, 0.695982668, 0.701952455,
                 0.707920713, 0.713885912]

LAD_4500_flow = [30.40501077, 31.15657663, 31.96620486, 33.00793503, 34.16534744,
                 35.84369869, 37.05873099, 38.44750789, 39.89405217, 40.81965762,
                 41.57166596, 41.97618509, 42.20681213, 42.32131444, 42.14668489,
                 41.79801575, 41.50770396, 41.04379505, 40.63750602, 40.28927936,
                 39.7671606, 39.07085476, 38.37484391, 37.85287264, 37.21448167,
                 36.63400557, 36.05338197, 35.58858812, 35.12364677, 34.71676778,
                 34.36750868, 33.84435748, 33.26329141, 32.62386801, 32.27431392,
                 31.635038, 31.22771654, 30.76233272, 30.06514194, 29.54169576,
                 28.90256733, 28.20508156, 27.85537999, 27.62121316, 27.4451087,
                 27.44274884, 27.26723434, 27.03424745, 26.74364067, 26.39497153,
                 25.81434794, 25.40776394, 24.94341255, 24.53697604, 24.13083451,
                 23.8979951, 23.89681517, 23.89563524, 24.18432464, 24.64661114,
                 25.34085211, 26.03494558, 26.96069852, 27.88645146, 28.92818163,
                 29.91199693, 30.40501077]
LAD_4500_func = interp1d(LAD_4500_time,LAD_4500_flow,fill_value='extrapolate',kind='quadratic')
"""
LAD_66_7_time = [0, 0.01132623, 0.020844545, 0.025610586, 0.030378156,
                 0.036338766, 0.041109396, 0.047066947, 0.054216008, 0.060181207,
                 0.067328738, 0.07447627, 0.080432291, 0.087581352, 0.093535844,
                 0.101865709, 0.114944789, 0.128017751, 0.136332321, 0.144642301,
                 0.15176536, 0.160073811, 0.169570713, 0.176690713, 0.186187615,
                 0.194491478, 0.20398838, 0.215859125, 0.22773293, 0.239608263,
                 0.251482068, 0.263358931, 0.274045814, 0.288301109, 0.300174914,
                 0.314428679, 0.331059347, 0.348873878, 0.364313035, 0.382130625,
                 0.397569782, 0.415382783, 0.432011922, 0.44863953, 0.464077159,
                 0.481891689, 0.50327922, 0.522294438, 0.540119675, 0.556753402,
                 0.574569462, 0.590007091, 0.60544166, 0.619687778, 0.63037466,
                 0.645821466, 0.657707506, 0.668409685, 0.675551098, 0.682695571,
                 0.688653122, 0.694607613, 0.700565164, 0.706528833, 0.712489443,
                 0.719641564, 0.724415252, 0.730388098, 0.737544807, 0.744703046,
                 0.751856695, 0.759010345, 0.767353976, 0.774503037, 0.782840549,
                 0.789985021, 0.799498749, 0.811387848, 0.822082379, 0.831583869,
                 0.837516948, 0.845828458, 0.852948458, 0.86244383, 0.869562301,
                 0.879059203, 0.888553046, 0.898049948, 0.908735301, 0.921799086,
                 0.936051323, 0.951488951, 0.964552736, 0.977621109, 0.988309522,
                 1.002557169, 1.015620954, 1.031061641, 1.050064623, 1.063126878,
                 1.080942938, 1.09994592, 1.118950431, 1.137953412, 1.1533941,
                 1.166456355, 1.184272415, 1.202096123, 1.21872832, 1.236550498,
                 1.256749579, 1.271003344, 1.287627894, 1.301875541, 1.316124719,
                 1.330375425, 1.34463225, 1.357706742, 1.366033547, 1.374366471,
                 1.381512473, 1.388660004, 1.398185968]

LAD_66_7_flow = [23.6289712, 23.85945074, 24.26367489, 24.72640387, 25.24704772,
                 25.94128868, 26.57776226, 27.15617349, 27.85026696, 28.71825252,
                 29.35443113, 29.99060973, 30.51110609, 31.20519956, 31.66778104,
                 32.07215268, 32.30218975, 32.30056735, 32.12579031, 31.77726866,
                 31.48680937, 31.08037286, 30.67378885, 30.26749983, 29.86091582,
                 29.2807347, 28.8741507, 28.35144197, 27.94456298, 27.59559886,
                 27.18871987, 26.89767062, 26.54885399, 26.31542463, 25.90854564,
                 25.6172014, 25.32556219, 24.86003087, 24.45270941, 24.10300784,
                 23.69568637, 23.17224019, 22.82268611, 22.41521716, 21.94998083,
                 21.48444951, 21.30805007, 21.30569021, 21.24556297, 21.06975349,
                 20.66213705, 20.19690072, 19.61583465, 19.03491608, 18.68609945,
                 18.56835233, 18.62479228, 18.85512434, 19.25964346, 19.77999233,
                 20.35840356, 20.82098504, 21.39939627, 22.20946697, 22.90370793,
                 23.71363113, 24.46593445, 25.62349436, 26.60716217, 27.64874485,
                 28.51658292, 29.384421, 30.31002645, 31.00411992, 31.6980659,
                 32.21841476, 32.44889431, 32.621164, 32.56192171, 32.32908231,
                 31.98085565, 31.69024887, 31.28395984, 30.81946097, 30.35525708,
                 29.94867307, 29.42625933, 29.01967532, 28.61294383, 28.26383221,
                 27.91457311, 27.44933678, 27.10022517, 26.92485816, 26.6339564,
                 26.11095269, 25.76184108, 25.41243449, 24.94675569, 24.53972921,
                 24.13211276, 23.66643396, 23.25867002, 22.79299122, 22.44358463,
                 22.03655815, 21.6289417, 21.5108996, 21.27717525, 21.10121828,
                 20.92496632, 20.63362209, 20.1103234, 19.58731969, 19.12223085,
                 18.71505688, 18.53954238, 18.59583485, 18.88437675, 19.40457813,
                 19.98284186, 20.61902046, 21.31281895]
"""
LAD_66_7_time = [0, 0.01132623, 0.020844545, 0.025610586, 0.030378156,
                 0.036338766, 0.041109396, 0.047066947, 0.054216008, 0.060181207,
                 0.067328738, 0.07447627, 0.080432291, 0.087581352, 0.093535844,
                 0.101865709, 0.114944789, 0.128017751, 0.136332321, 0.144642301,
                 0.15176536, 0.160073811, 0.169570713, 0.176690713, 0.186187615,
                 0.194491478, 0.20398838, 0.215859125, 0.22773293, 0.239608263,
                 0.251482068, 0.263358931, 0.274045814, 0.288301109, 0.300174914,
                 0.314428679, 0.331059347, 0.348873878, 0.364313035, 0.382130625,
                 0.397569782, 0.415382783, 0.432011922, 0.44863953, 0.464077159,
                 0.481891689, 0.50327922, 0.522294438, 0.540119675, 0.556753402,
                 0.574569462, 0.590007091, 0.60544166, 0.619687778, 0.63037466,
                 0.645821466, 0.657707506, 0.668409685, 0.675551098, 0.682695571,
                 0.688653122, 0.694607613, 0.700565164, 0.706528833, 0.712489443,
                 0.719641564]

LAD_66_7_flow = [23.6289712, 23.85945074, 24.26367489, 24.72640387, 25.24704772,
                 25.94128868, 26.57776226, 27.15617349, 27.85026696, 28.71825252,
                 29.35443113, 29.99060973, 30.51110609, 31.20519956, 31.66778104,
                 32.07215268, 32.30218975, 32.30056735, 32.12579031, 31.77726866,
                 31.48680937, 31.08037286, 30.67378885, 30.26749983, 29.86091582,
                 29.2807347, 28.8741507, 28.35144197, 27.94456298, 27.59559886,
                 27.18871987, 26.89767062, 26.54885399, 26.31542463, 25.90854564,
                 25.6172014, 25.32556219, 24.86003087, 24.45270941, 24.10300784,
                 23.69568637, 23.17224019, 22.82268611, 22.41521716, 21.94998083,
                 21.48444951, 21.30805007, 21.30569021, 21.24556297, 21.06975349,
                 20.66213705, 20.19690072, 19.61583465, 19.03491608, 18.68609945,
                 18.56835233, 18.62479228, 18.85512434, 19.25964346, 19.77999233,
                 20.35840356, 20.82098504, 21.39939627, 22.20946697, 22.90370793,
                 23.6289712]
LAD_66_7_func = interp1d(LAD_66_7_time,LAD_66_7_flow,fill_value='extrapolate',kind='quadratic')
"""
LAD_22_2_time = [0, 0.015764949, 0.027660167, 0.037181542, 0.046699858,
                 0.057409684, 0.065744137, 0.076453963, 0.087162259, 0.100247457,
                 0.112136557, 0.125211049, 0.137092501, 0.150160875, 0.162040797,
                 0.171540758, 0.183419151, 0.195294485, 0.205982898, 0.21667131,
                 0.230928135, 0.243994979, 0.257063352, 0.270133256, 0.28439008,
                 0.298645375, 0.314090651, 0.328347476, 0.346169654, 0.362801852,
                 0.38062403, 0.397257757, 0.416265327, 0.432897524, 0.453095075,
                 0.475667998, 0.494678628, 0.516067689, 0.537459809, 0.560037321,
                 0.577859499, 0.600429363, 0.624187679, 0.642011386, 0.662215055,
                 0.674104155, 0.68243096, 0.690759295, 0.699087631, 0.707420554,
                 0.716941929, 0.726466363, 0.734803875, 0.744328309, 0.755042723,
                 0.764565628, 0.774083943, 0.78717373, 0.80025434, 0.81214344,
                 0.825214873, 0.837093266, 0.851350091, 0.862038503, 0.872726915,
                 0.884603779, 0.897666034, 0.910732878, 0.926176625, 0.940433449,
                 0.955877196, 0.973699374, 0.992710003, 1.012907554, 1.029538222,
                 1.046170419, 1.062802616, 1.085378599, 1.107949993, 1.129334465,
                 1.151908918, 1.169728037, 1.194680922, 1.22320069, 1.249342025,
                 1.273106459, 1.29330248, 1.314683894, 1.334881444, 1.347952877,
                 1.36221582, 1.375297959, 1.386003197, 1.399091454]

LAD_22_2_flow = [11.6405935, 11.92839795, 12.33232712, 12.852381, 13.25660515,
                 13.77651154, 14.35462778, 14.87453417, 15.3365257, 15.79822224,
                 15.97049193, 16.0267844, 15.90947975, 15.73411274, 15.55889323,
                 15.26813896, 15.03500457, 14.68604045, 14.39513869, 14.10423693,
                 13.92872243, 13.69544056, 13.52007355, 13.40262141, 13.22710691,
                 12.99367754, 12.81801556, 12.64250106, 12.46654409, 12.23281974,
                 12.05686276, 11.88105328, 11.58911908, 11.35539474, 11.12122791,
                 10.82885124, 10.65274678, 10.5342622, 10.53160736, 10.41297529,
                 10.23701832, 9.828811909, 9.420458009, 9.302415905, 9.299908555,
                 9.472178248, 9.76072015, 10.10717692, 10.45363369, 10.97383507,
                 11.49388895, 12.12977257, 12.82371855, 13.45960217, 14.15325316,
                 14.73122192, 15.13544606, 15.77088721, 16.05883915, 16.23110884,
                 16.17157157, 15.93843718, 15.76292269, 15.47202092, 15.18111916,
                 14.89006991, 14.48304343, 14.24976156, 14.0161847, 13.8406702,
                 13.60709334, 13.43113637, 13.25503191, 13.02086508, 12.72922587,
                 12.49550152, 12.26177717, 12.08523023, 11.73493869, 11.44270951,
                 11.20824771, 10.916461, 10.73961908, 10.62024956, 10.44326014,
                 10.26656572, 9.974484027, 9.566425109, 9.332258287, 9.272721016,
                 9.32886599, 9.674732795, 10.02089458, 10.59842086]
"""
LAD_22_2_time = [0, 0.015764949, 0.027660167, 0.037181542, 0.046699858,
                 0.057409684, 0.065744137, 0.076453963, 0.087162259, 0.100247457,
                 0.112136557, 0.125211049, 0.137092501, 0.150160875, 0.162040797,
                 0.171540758, 0.183419151, 0.195294485, 0.205982898, 0.21667131,
                 0.230928135, 0.243994979, 0.257063352, 0.270133256, 0.28439008,
                 0.298645375, 0.314090651, 0.328347476, 0.346169654, 0.362801852,
                 0.38062403, 0.397257757, 0.416265327, 0.432897524, 0.453095075,
                 0.475667998, 0.494678628, 0.516067689, 0.537459809, 0.560037321,
                 0.577859499, 0.600429363, 0.624187679, 0.642011386, 0.662215055,
                 0.674104155, 0.68243096, 0.690759295, 0.699087631, 0.707420554,
                 0.716941929]

LAD_22_2_flow = [11.6405935, 11.92839795, 12.33232712, 12.852381, 13.25660515,
                 13.77651154, 14.35462778, 14.87453417, 15.3365257, 15.79822224,
                 15.97049193, 16.0267844, 15.90947975, 15.73411274, 15.55889323,
                 15.26813896, 15.03500457, 14.68604045, 14.39513869, 14.10423693,
                 13.92872243, 13.69544056, 13.52007355, 13.40262141, 13.22710691,
                 12.99367754, 12.81801556, 12.64250106, 12.46654409, 12.23281974,
                 12.05686276, 11.88105328, 11.58911908, 11.35539474, 11.12122791,
                 10.82885124, 10.65274678, 10.5342622, 10.53160736, 10.41297529,
                 10.23701832, 9.828811909, 9.420458009, 9.302415905, 9.299908555,
                 9.472178248, 9.76072015, 10.10717692, 10.45363369, 10.97383507,
                 11.6405935]
LAD_22_2_func = interp1d(LAD_22_2_time,LAD_22_2_flow,fill_value='extrapolate')
"""
LAD_16_1_time = [0, 0.01798278, 0.035815664, 0.052466216, 0.066736807,
                 0.081007398, 0.09765642, 0.115487775, 0.132126091, 0.153513622,
                 0.177278056, 0.198660999, 0.223613883, 0.250942141, 0.275895026,
                 0.303224813, 0.338872228, 0.373335781, 0.408981666, 0.448194435,
                 0.480279556, 0.513553128, 0.551578975, 0.593173234, 0.630006042,
                 0.657335829, 0.681112499, 0.701325345, 0.719164348, 0.737007939,
                 0.758416884, 0.778637378, 0.807170912, 0.839257562, 0.86658429,
                 0.891532586, 0.92480157, 0.959260534, 0.998473302, 1.041249895,
                 1.082839566, 1.12799459, 1.162452024, 1.20998701, 1.249202838,
                 1.291982489, 1.325249943, 1.359715025, 1.400134599]


LAD_16_1_flow = [5.791191827, 5.904956687, 6.134403792, 6.59565786, 6.941377174,
                 7.287096489, 7.690435689, 7.861967926, 7.859903049, 7.683503604,
                 7.506809176, 7.156665126, 6.979823207, 6.744771438, 6.567929519,
                 6.390792618, 6.15470841, 6.092516298, 5.798517222, 5.561990541,
                 5.442178543, 5.322219054, 5.143754732, 5.080677673, 4.728616238,
                 4.551479336, 4.838103853, 5.183085712, 5.644192289, 6.279043471,
                 6.913452178, 7.548008377, 7.949872666, 7.887975536, 7.595008899,
                 7.244422375, 6.950718282, 6.714781566, 6.478254884, 6.183370862,
                 5.946549198, 5.709285061, 5.415433477, 5.293704093, 5.173007148,
                 4.993952862, 4.6423339, 4.638056656, 5.096360901]
"""
LAD_16_1_time = [0, 0.01798278, 0.035815664, 0.052466216, 0.066736807,
                 0.081007398, 0.09765642, 0.115487775, 0.132126091, 0.153513622,
                 0.198660999, 0.223613883, 0.250942141, 0.275895026,
                 0.303224813, 0.338872228, 0.408981666, 0.448194435,
                 0.480279556, 0.513553128, 0.551578975,
                 0.657335829, 0.681112499, 0.701325345, 0.719164348]

LAD_16_1_flow = [5.791191827, 5.904956687, 6.134403792, 6.59565786, 6.941377174,
                 7.287096489, 7.690435689, 7.861967926, 7.859903049, 7.683503604,
                 7.156665126, 6.979823207, 6.744771438, 6.567929519,
                 6.390792618, 6.15470841, 5.798517222, 5.561990541,
                 5.442178543, 5.322219054, 5.143754732,
                 4.551479336, 4.838103853, 5.183085712, 5.791191827]
LAD_16_1_func = interp1d(LAD_16_1_time,LAD_16_1_flow,fill_value='extrapolate',kind='quadratic')

"""
LAD_9_time = [0, 0.027369555, 0.052328558, 0.079667522, 0.110573369,
                 0.146228432, 0.183065828, 0.225848539, 0.261502072, 0.309033998,
                 0.349442866, 0.394602479, 0.433821365, 0.488485528, 0.536025102,
                 0.579997793, 0.629908151, 0.673880842, 0.719046573,
              0.760653069,
              0.80582033, 0.858110649, 0.910397909, 0.962689758, 1.023297706,
              1.087469478, 1.140949778, 1.193238568, 1.25028575, 1.297823795,
              1.353677938, 1.40003365]


LAD_9_flow = [1.331746984, 1.328502178, 1.383319731, 1.553672039, 1.78149674,
              1.834986873, 1.656670043, 1.593445492, 1.589020757, 1.351461638,
              1.404361806, 1.340842273, 1.335975064, 1.155445866, 1.207461087,
              1.202003913, 1.02206468, 1.016607506, 1.184747446,
              1.584989331,
              1.811044139, 1.746639659, 1.566405443, 1.559915832, 1.436564046,
              1.254854918, 1.248217815, 1.125898467, 1.176733759, 1.170834112,
              1.04807229, 1.273979607]
"""
# Selected Data for Smooth interpolation
LAD_9_time = [   0,           0.079667522, 0.110573369,
                 0.146228432, 0.261502072,
                 0.394602479, 0.536025102,
                 0.629908151, 0.673880842, 0.719046573]

LAD_9_flow = [1.331746984, 1.553672039, 1.78149674,
              1.834986873, 1.589020757,
              1.340842273, 1.207461087,
              1.02206468, 1.016607506, 1.331746984]

LAD_9_func = interp1d(LAD_9_time,LAD_9_flow,fill_value='extrapolate',kind='quadratic')


LAD_time  = [LAD_4500_time,LAD_66_7_time,LAD_22_2_time,LAD_16_1_time,LAD_9_time]
LAD_flow  = [LAD_4500_flow,LAD_66_7_flow,LAD_22_2_flow,LAD_16_1_flow,LAD_9_flow]
LAD_funcs = [LAD_4500_func,LAD_66_7_func,LAD_22_2_func,LAD_16_1_func,LAD_9_func]
LAD_diam  = [0.45,0.00667,0.00222,0.00161,0.0009]
def generate_physiologic_wave(flow_value,diameter,time=LAD_time,
                              flow=LAD_flow,normalize_time=False,
                              one_cycle=True,n_time_points=50,min_buffer=0.05,
                              pulse_by_diameter=False):
    LAD_diam = [0.45,0.00667,0.00222,0.00161,0.0009] #in cm units
    LAD_amp_d  = [abs(max(flow[0])-min(flow[0]))/60,abs(max(flow[1])-min(flow[1]))/60,
                abs(max(flow[2])-min(flow[2]))/60,abs(max(flow[3])-min(flow[3]))/60,
                abs(max(flow[4])-min(flow[4]))/60]
    LAD_amp_ff  = [(np.mean(flow[0])/60),(np.mean(flow[1])/60),(np.mean(flow[2])/60),
                   (np.mean(flow[3])/60),(np.mean(flow[4])/60)]
    LAD_amp_f  = [LAD_amp_d[0]/(np.mean(flow[0])/60),LAD_amp_d[1]/(np.mean(flow[1])/60),LAD_amp_d[2]/(np.mean(flow[2])/60),
                  LAD_amp_d[3]/(np.mean(flow[3])/60),LAD_amp_d[4]/(np.mean(flow[4])/60)]

    amp_d = interp1d(LAD_diam,LAD_amp_d,fill_value='extrapolate')
    amp_f = interp1d(LAD_amp_ff,LAD_amp_f,fill_value='extrapolate')
    for i,vessel in enumerate(time):
        time[i] = np.array(vessel)
    for i,vessel in enumerate(flow):
        flow[i] = np.array(vessel)

    if one_cycle:
        t = np.linspace(0,time[0][-1]/2,num=200)
    else:
        t = np.linspace(0,time[0][-1],num=200)

    flow_interp = []
    time_interp = []
    diam_interp = []
    interp = []

    for i,_ in enumerate(time):
        f = interp1d(time[i],flow[i],fill_value='extrapolate')
        flow_tmp = f(t)
        interp.append(f)
        diam_interp.extend(np.ones(len(flow_tmp))*LAD_diam[i])
        flow_interp.extend(flow_tmp)
        time_interp.extend(t)

    #for i in range(len(t)):
    #    fd = interp1d(LAD_diam,[flo[i] for flo in flow_interp],fill_value='extrapolate')
    diam_interp = np.array(diam_interp)
    flow_interp = np.array(flow_interp)
    time_interp = np.array(time_interp)

    ND_interp = LinearNDInterpolator(list(zip(time_interp,diam_interp)),flow_interp)
    KND_interp = NearestNDInterpolator(list(zip(time_interp,diam_interp)),flow_interp)
    t = np.linspace(0,t[-1],num=n_time_points)
    d = np.ones(len(t))*diameter
    wave = ND_interp(t,d)/60
    if np.any(np.isnan(wave)):
        wave = KND_interp(t,d)/60
        print(wave)
        if pulse_by_diameter:
            post_wave_amp = amp_d(diameter)
            pre_wave_min = np.min(wave)
            pre_wave_max = np.max(wave)
            pre_wave_amp = pre_wave_max - pre_wave_min
            wave_amp_scale = post_wave_amp/pre_wave_amp
            wave = wave*wave_amp_scale
        else:
            wave_amp_ratio = amp_f(flow_value)
            print("ratio: {}".format(wave_amp_ratio))
            pre_wave_min = np.min(wave)
            pre_wave_max = np.max(wave)
            pre_wave_amp = pre_wave_max - pre_wave_min
            pre_amp_ratio = pre_wave_amp/flow_value
            #wave_ratio_scale = wave_amp_ratio/pre_amp_ratio
            post_wave_amp = flow_value*wave_amp_ratio
            wave_amp_scale = post_wave_amp/pre_wave_amp
            wave = wave*wave_amp_scale
    #wave_max = np.max(wave)
    wave_min = np.min(wave)
    wave_mean = np.mean(wave)
    wave_diff = wave - wave_mean
    wave_diff_min = np.min(wave_diff)
    if (flow_value + wave_diff_min) < 0:
        wave_return = abs(wave_diff_min) + wave_diff + flow_value*min_buffer
    else:
        wave_return = flow_value + wave_diff
    return t, wave_return

def wave(mean_flow,diameter,time=LAD_time,
         flow=LAD_flow, diam=LAD_diam,n_time_points=50):
    funcs = []
    for i,_ in enumerate(time):
        time[i] = np.array(time[i])/time[i][-1]
        flow[i] = np.array(flow[i])
        diam[i] = np.array(diam[i])
        funcs.append(interp1d(time[i],flow[i],fill_value='extrapolate',kind='quadratic'))

    for i,_ in enumerate(time):
        tmp_time  = np.linspace(0,1,n_time_points)
        tmp_diams = diam[i]*np.ones((n_time_points,1))
        tmp_coor  = np.hstack((tmp_time.reshape(-1,1),tmp_diams))
        tmp_flow  = funcs[i](tmp_time).reshape(-1,1)
        if i == 0:
            COOR = tmp_coor
            FLOW = tmp_flow
        else:
            COOR = np.vstack((COOR,tmp_coor))
            FLOW = np.vstack((FLOW,tmp_flow))
    if diameter > LAD_diam[0] or diameter < LAD_diam[-1]:
        interp = NearestNDInterpolator(list(zip(COOR[:,0].flatten(),COOR[:,1].flatten())),FLOW.flatten())
    else:
        interp = LinearNDInterpolator(list(zip(COOR[:,0].flatten(),COOR[:,1].flatten())),FLOW.flatten())
    d_values = diameter*np.ones(n_time_points)
    t_values = np.linspace(0,1,n_time_points)
    f_values = interp(t_values,d_values)
    amp      = (max(f_values) - min(f_values))/2
    f_mean   = np.mean(f_values)
    if mean_flow > amp:
        f_values = f_values - f_mean + mean_flow
    else:
        print('WARNING: Shifting Mean flow')
        shift = abs(amp-mean_flow)
        mean_flow = mean_flow + shift*1.1
        f_values = f_values - f_mean + mean_flow
    return t_values,f_values
