import numpy as np   # Needed for arrays
import scipy.special # Needed for binomial combinations
import scipy.stats   # Needed for Poisson distributions
import itertools     # Needed for Q iteration in STRs
import random        # Needed for occasionally randomising inputs
import csv           # Needed for output files
from math import log10
from math import sqrt

# =============================================
# Functions defining random generation of lines
# =============================================

def generations(tmrca):
    g=100
    n=np.zeros(g)
    for j in np.arange(100):
        a=np.random.normal(loc=35,scale=8,size=g)
        for i in np.arange(len(a)):
            b=np.sum(a[0:i-1])
            c=np.sum(a[0:i])
            if (b-tmrca<0 and c-tmrca>0):
                if (abs(b-tmrca)<abs(c-tmrca)):
                    n[i-1]+=1
                else:
                    n[i]+=1
    n/=np.sum(n)
    choice=np.ndarray.flatten(np.random.choice(np.arange(len(a)), 1, p=n))
    return choice

def strs(gens):
    mustr=np.array([0.001776,0.003558,0.002262963333333,0.00214183,0.001832,0.003458,0.0002157006,0.000537249333333,0.003600416666667,0.002361616666667,0.000516113666667,0.0026555,0.007048133333333,0.000523,0.001525,0.000868596333333,0.000918865,0.0019482,0.001313576666667,0.001448,0.0089895,0.001874,0.0032,0.004241,0.003716,0.002626356666667,0.0022795,0.000496,0.001141,0.004139643333333,0.002660216666667,0.0096684,0.006975733333333,0.014357,0.018449,0.002620776666667,0.000612560666667,0.001436,0.000990948666667,0.000433,0.000319,0.000254032,0.00166647,0.000919199333333,0.0002167,0.00282426,0.0017663,0.000236,0.00229,0.001527,0.0034075,0.000467199,0.000203582333333,0.0003603,0.005522,0.000264693666667,0.0027805,0.005328563333333,0.0020715,0.0028765,0.00055675,0.001144006333333,0.00081927,0.001894,0.000297,0.000300432666667,0.001237244666667,0.018279,0.00106261,6.96637E-05,0.000826361,0.001597966666667,0.007726,0.001002,0.00064516,0.00218298,0.001634593333333,0.004000833333333,0.0007529,0.002278673333333,0.000260055333333,0.0027318,0.001357,0.000812522666667,0.0013825,0.001339923333333,0.00105415,0.001369627,0.00373403,0.00076,0.002423076666667,0.00084,0.00178172,0.016378,0.000278238,0.007583,0.004142,0.00387799,0.006949,0.0024505,0.00176344,0.00301,0.0002895845,0.003112,0.0008633,0.00133475,0.0007121,0.0026829,0.001059957666667,0.002533206666667,0.000909901666667,1.00E-06,1.00E-06,1.00E-06,1.00E-06,1.00E-06,1.00E-06,1.00E-06,0.000866329393708,0.000646939655185,1.00E-06,1.00E-06,1.00E-06,1.00E-06,0.000185879225554,1.00E-06,1.00E-06,1.00E-06,7.18628072330345E-06,1.00E-06,1.00E-06,1.00E-06,1.00E-06,1.00E-06,1.00E-06,1.00E-06,1.00E-06,1.00E-06,1.00E-06,1.00E-06,1.00E-06,4.30389844826262E-05,1.00E-06,1.00E-06,1.00E-06,1.00E-06,1.00E-06,1.00E-06,1.00E-06,1.00E-06,4.31170729064071E-05,1.00E-06,1.00E-06,1.00E-06,1.00E-06,1.00E-06,1.00E-06,1.00E-06,1.00E-06,5.77076443913719E-05,7.19120620701156E-06,1.00E-06,1.00E-06,1.00E-06,2.88041462217885E-05,1.00E-06,7.17644996995205E-06,1.00E-06,1.00E-06,1.00E-06,1.00E-06,7.20107745785662E-06,7.19613844723963E-06,1.00E-06,1.00E-06,1.00E-06,0.000230751008629,1.00E-06,1.43627104792525E-05,6.48782185068035E-05,7.22090148387681E-06,1.00E-06,6.50328710086277E-05,8.6689637041137E-05,1.00E-06,0.001323535604723,1.00E-06,1.00E-06,1.00E-06,1.00E-06,2.16477441055493E-05,7.20602325276963E-06,1.00E-06,1.00E-06,1.43823988732946E-05,1.00E-06,1.00E-06,0.008133689075989,1.00E-06,1.00E-06,1.00E-06,1.00E-06,1.00E-06,1.00E-06,1.00E-06,1.00E-06,1.00E-06,1.44120328808033E-05,0.000178671921343,1.00E-06,1.44716687654042E-05,1.00E-06,7.20107745785662E-06,8.62299433967952E-05,4.33247885674273E-05,1.00E-06,1.00E-06,2.90634603345777E-05,1.00E-06,0.003577001096869,1.00E-06,1.00E-06,1.00E-06,1.00E-06,1.00E-06,7.230854483894E-06,7.21593525175012E-06,1.00E-06,1.00E-06,7.34217624636386E-06,1.00E-06,1.00E-06,7.32679459221162E-06,7.26590710251738E-06,1.00E-06,1.00E-06,1.00E-06,1.00E-06,0.000340653719367,1.00E-06,7.49426418290361E-06,1.00E-06,8.71104600989299E-05,7.23584128005022E-06,7.23085448384135E-06,1.00E-06,7.26087879312772E-06,1.00E-06,1.00E-06,1.00E-06,1.00E-06,0.000107848004839,1.00E-06,0.00070976761243,0.002858815116904,1.00E-06,2.16031711042177E-05,1.00E-06,7.25084302421095E-06,7.311477251621E-06,0.001408707269388,1.00E-06,1.00E-06,0.001773519683037,1.00E-06,0.000115613331671,1.00E-06,1.00E-06,0.000156930073995,1.00E-06,1.00E-06,1.00E-06,2.16180083626181E-05,1.00E-06,2.16626426743719E-05,7.21097584608153E-06,0.000264790198753,6.55725075769289E-05,1.4501672167711E-05,0.000957091249245,1.00E-06,1.00E-06,7.23584128005022E-06,0.004884093197363,5.80669449264951E-05,7.23085448384135E-06,0.000375542091327,9.32869226725564E-05,7.39913247958075E-06,1.00E-06,2.89234179353917E-05,1.00E-06,1.45822928440098E-05,0.000240161874603,1.00E-06,1.45318142050708E-05,2.2098698583111E-05,0.00054974441262,1.00E-06,1.00E-06,1.00E-06,0.000814739188919,0.001692263028863,7.29115347885296E-06,1.00E-06,0.00029993372441,0.000151842964233,1.00E-06,3.04115068291627E-05,1.00E-06,0.002179366480775,1.00E-06,0.001176200984885,1.00E-06,3.62038277773263E-05,0.000228878374873,1.00E-06,1.00E-06,7.28609017780372E-06,0.000695661057329,0.002038775146703,0.00034583869351,2.17325060312092E-05,0.000202042136976,2.90837695247712E-05,0.000224370527201,1.00E-06,2.91039385743883E-05,1.00E-06,7.26892826973318E-05,1.00E-06,7.2962238220587E-06,1.00E-06,0.000479061056326,1.00E-06,7.21097584608153E-06,1.00E-06,0.000345084936576,1.00E-06,6.48194321241704E-05,1.00E-06,1.00E-06,1.00E-06,1.00E-06,4.3275773887849E-05,1.00E-06,1.00E-06,1.00E-06,1.00E-06,1.00E-06,1.00E-06,2.22078408156419E-05,1.00E-06,1.00E-06,7.311477251621E-06,0.000137443809509,1.00E-06,1.00E-06,7.62497809308834E-06,0.000228722497731,2.17224425871507E-05,1.4501672167711E-05,2.31383827594682E-05,2.17025370439257E-05,1.45924335008614E-05,1.00E-06,2.2514887404072E-05,1.00E-06,0.000727471884019,0.004627006466369,0.000908822480141,1.00E-06,1.00E-06,1.00E-06,6.13784627205385E-05,1.00E-06,1.00E-06,0.000253104032158,1.00E-06,1.00E-06,1.00E-06,0.005655396000866,1.00E-06,0.002290119886728,1.00E-06,1.00E-06,0.000141953388767,1.00E-06,0.002976316037582,0.000794431508604,7.32168168604791E-06,1.00E-06,0.000120857824116,1.00E-06,0.001113326336591,1.00E-06,1.00E-06,0.000152916056261,1.00E-06,0.005242230390096,1.00E-06,2.34661382616262E-05,7.26590710251738E-06,2.23708698150525E-05,1.00E-06,4.73440664708861E-05,1.00E-06,1.00E-06,1.00E-06,1.00E-06,1.00E-06,0.000327325431358,0.000122996751763,1.00E-06,1.00E-06,8.73612992481482E-05,0.00153550001202,2.21921831100369E-05,7.24083495932796E-06,0.000376992758557,1.00E-06,1.00E-06,2.16577382194611E-05,8.10249390867033E-05,0.008714410639185,0.000829703192635,0.000872147285301,7.60839003340416E-06,1.00E-06,0.004355995076024,2.28914976825357E-05,0.001977045000582,0.002592997087187,1.00E-06,1.00E-06,2.21609338427711E-05,1.52057374107413E-05,0.004876561421801,0.001447751657171,0.001501338426772,7.43583972788376E-06,0.000896489821155,1.00E-06,1.00E-06,7.19613844723963E-06,1.00E-06,1.00E-06,1.00E-06,3.62842807279286E-05,1.00E-06,1.00E-06,1.00E-06,7.3887111662081E-06,1.00E-06,0.000125529750045,1.00E-06,1.00E-06,0.000129422649816,0.001287373752605,0.001351076257623,1.00E-06,3.63040440504395E-05,7.32679459221162E-06,0.000123344245185,7.22476883737539E-05,1.00E-06,7.39391815085335E-06,7.87685424626355E-06,0.000159909999226,1.00E-06,7.38351151023684E-06,1.00E-06,3.20853084707844E-05,0.000701370982279,7.75788957522413E-05,7.33704185737334E-06,7.4782393842272E-06,7.40435416800369E-06,0.000994525963611,1.00E-06,7.06549683294134E-05,1.00E-06,0.002030114945391,1.53728496059504E-05,2.99129575368543E-05,0.000473332555761,1.00E-06,0.001022672298961,0.000323892700767,0.00051820765804,0.002086720643268,2.91443607112809E-05,1.00E-06,0.000195652693676,7.69770348940673E-06,1.00E-06,0.000651357389689,0.001469025770727,1.00E-06,5.36365594267711E-05,1.00E-06,0.001009358042687,1.00E-06,0.00093988518356,2.30761053423314E-05,0.002999455874981,1.00E-06,0.001095112066371,1.00E-06,1.00E-06,3.55958566255592E-05,1.00E-06,1.00E-06,1.00E-06,1.00E-06,0.000188971689956,0.001242269874803,1.60182558783611E-05,1.00E-06,4.78348932246706E-05,0.00038676857562,1.00E-06,2.59273318752663E-05,1.53391372164636E-05,7.54275331133574E-06,0.006006121688146,1.00E-06,1.00E-06,1.00E-06,0.002345691789981,1.00E-06,1.00E-06,0.002382318331383,1.00E-06,7.3700230952696E-05,0.000391849483953,1.00E-06,0.001089617317705,0.000111346867817,0.000802564339049,0.000153930897162,1.00E-06,0.000107485260618,1.00E-06,1.00E-06,1.52167800668037E-05,0.000721185331166,6.08004789120523E-05,0.000561307787544,2.25903683039772E-05,1.50207156135449E-05,1.00E-06,0.001393079169983,0.001079690824854,1.00E-06,5.4662827803081E-05,0.000910753943082,1.00E-06,0.00068480347357,0.001227366055884,0.000263443029256])
    strnames=["DYS393","DYS390","DYS19","DYS391","DYS385a","DYS385b","DYS426","DYS388","DYS439","DYS389I","DYS392","DYS389II","DYS458","DYS459a","DYS459b","DYS455","DYS454","DYS447","DYS437","DYS448","DYS449","DYS464a","DYS464b","DYS464c","DYS464d","DYS460","Y-GATA-H4","YCAIIa","YCAIIb","DYS456","DYS607","DYS576","DYS570","CDYa","CDYb","DYS442","DYS438","DYS531","DYS578","DYF395S1a","DYF395S1b","DYS590","DYS537","DYS641","DYS472","DYF406S1","DYS511","DYS425","DYS413a","DYS413b","DYS557","DYS594","DYS436","DYS490","DYS534","DYS450","DYS444","DYS481","DYS520","DYS446","DYS617","DYS568","DYS487","DYS572","DYS640","DYS492","DYS565","DYS710","DYS485","DYS632","DYS495","DYS540","DYS714","DYS716","DYS717","DYS505","DYS556","DYS549","DYS589","DYS522","DYS494","DYS533","DYS636","DYS575","DYS638","DYS462","DYS452","DYS445","Y-GATA-A10","DYS463","DYS441","Y-GGAAT-1B07","DYS525","DYS712","DYS593","DYS650","DYS532","DYS715","DYS504","DYS513","DYS561","DYS552","DYS726","DYS635","DYS587","DYS643","DYS497","DYS510","DYS434","DYS461","DYS435","FTY371","FTY303","FTY10","FTY284","FTY219","FTY324","FTY327","FTY394","DYS538","FTY220","FTY254","FTY19","FTY4","DYS577","FTY24","FTY346","FTY160","FTY62","FTY400","FTY114","FTY246","FTY26","FTY368","FTY173","FTY209","FTY337","FTY81","FTY387","FTY361","FTY330","FTY65","FTY168","FTY171","FTY136","FTY53","FTY40","FTY386","FTY100","FTY152","FTY55","FTY297","FTY370","FTY103","FTY64","FTY174","FTY313","FTY362","FTY342","FTY161","FTY374","FTY316","FTY101","FTY204","DYS477","FTY192","DYS502","FTY341","DYS493","FTY1","FTY227","DYS499","FTY391","FTY353","FTY332","FTY163","DYS483","FTY172","FTY180","DYS581","FTY179","FTY188","FTY369","FTY70","FTY63","DYS508","FTY347","FTY83","FTY277","FTY276","FTY195","FTY109","FTY356","FTY215","FTY105","FTY123","FTY242","FTY214","FTY132","FTY262","FTY333","FTY208","DYF398B","FTY43","FTY166","FTY13","FTY11","DYS584","DYS608","FTY95","FTY151","FTY388","FTY256","DYS580","FTY234","FTY268","FTY139","FTY27","FTY75","DYS512","FTY329","FTY320","FTY7","FTY247","FTY211","DYS474","FTY39","FTY376","FTY37","FTY373","DYS475","FTY138","FTY288","FTY264","FTY93","FTY380","FTY18","FTY115","DYS569","FTY390","FTY46","FTY243","FTY322","FTY281","FTY153","FTY181","FTY184","FTY359","FTY121","FTY74","FTY279","DYS530","FTY45","DYS573","DYS542","FTY36","FTY304","FTY203","FTY291","FTY142","FTY191","FTY183","FTY141","FTY299","FTY193","FTY124","FTY16","FTY236","FTY185","FTY378","FTY225","FTY397","FTY3","FTY67","FTY182","FTY334","DYS623","FTY348","FTY357","FTY275","FTY306","FTY383","FTY253","FTY285","DYS645","FTY17","FTY85","DYS598","FTY375","FTY325","FTY12","FTY365","FTY35","FTY237","FTY238","FTY305","FTY56","FTY129","FTY352","FTY292","FTY154","DYS539","FTY301","FTY366","FTY231","DYS618","FTY216","FTY82","FTY393","FTY155","FTY68","FTY32","FTY265","FTY300","FTY143","FTY201","FTY199","FTY86","FTY98","FTY144","DYS541","DYS507","FTY20","FTY116","FTY308","FTY177","FTY89","FTY158","FTY22","FTY84","FTY186","FTY396","FTY91","FTY29","FTY78","FTY229","FTY260","FTY250","FTY364","DYS476","FTY221","FTY57","FTY377","FTY384","DYS466","FTY363","FTY137","FTY69","FTY72","FTY251","FTY2","FTY217","FTY257","FTY120","FTY25","FTY252","FTY33","FTY42","FTY54","FTY372","DYS480","FTY176","FTY76","FTY66","FTY197","FTY385","FTY270","FTY298","FTY198","FTY117","DYS544","FTY159","FTY232","FTY156","FTY343","FTY296","FTY311","FTY344","FTY178","FTY367","FTY73","FTY112","FTY295","DYS616","FTY338","FTY196","FTY157","FTY162","DYS551","FTY148","FTY94","FTY52","FTY131","FTY9","FTY194","FTY8","FTY318","FTY317","FTY99","FTY289","FTY169","FTY51","FTY345","FTY398","FTY111","FTY273","FTY382","FTY47","FTY30","FTY147","DYS615","FTY302","FTY354","FTY326","FTY88","FTY248","FTY267","FTY170","FTY60","DYS453","FTY249","FTY167","DYS624","FTY134","FTY235","FTY145","FTY14","FTY80","FTY50","DYS514","FTY340","DYS585","DYS516","FTY293","FTY207","FTY278","FTY210","DYS523","FTY269","FTY530","FTY1156","FTY1070","FTY906","FTY1004","FTY331","FTY31","FTY226","FTY335","DYS583","FTY339","FTY113","FTY280","FTY255","FTY59","DYS620","FTY349","FTY130","FTY119","DYF398A","FTY58","FTY127","FTY321","FTY92","FTY392","DYF392","FTY290","FTY312","FTY135","FTY150","FTY294","FTY244","FTY48","FTY282","DYS489","FTY272","FTY239","FTY258","FTY49","DYS574","FTY41","FTY905","FTY1103","FTY502","FTY743","FTY443","FTY670","FTY883","DYS631","FTY1042","FTY510","DYS389B","FTY512","FTY837","DYS642","FTY407","FTY935","FTY563","FTY1016","FTY1091","FTY1049","FTY1155","DYS602","FTY587","FTY635","FTY904","DYS543","FTY1148","FTY861","FTY742","FTY433","FTY1039","FTY71","FTY259","FTY108","FTY336","FTY44","FTY233","FTY274","FTY34","FTY942","FTY1068","FTY945","FTY1051","FTY658","FTY625","FTY800","FTY818","FTY971","FTY509","FTY1025","FTY835","FTY984","DYS637","FTY832","FTY689","FTY998","FTY520","FTY445","FTY809","FTY428","FTY562","FTY897","FTY1064","FTY1088","FTY1052","DYF405","FTY310","FTY189","DYS488","FTY283","FTY28","FTY578","DYS484","FTY858","FTY596","FTY915","FTY446","FTY452","FTY1006","FTY612","FTY507","FTY614","FTY824","FTY1055","FTY690","FTY2318","FTY981","FTY946","FTY1040","DYS718","FTY15","FTY2025","DYS685u1","DYS596u5","FTY1511","FTY769","FTY923","FTY1167","FTY415","FTY505","FTY241","FTY2263","FTY517","FTY2180","FTY775","DYS567","FTY444","FTY691","FTY1120","FTY223","DYF393u3","FTY1850","FTY645","FTY696","FTY1166","FTY991","FTY912","DYS551u4","FTY447","DYS518u3","FTY875","DYS536u1","FTY585","FTY896","FTY731","FTY401","FTY2011","FTY665","FTY814","FTY2050","DYS626u3","FTY750","FTY471","FTY985","FTY589","FTY420","FTY921","FTY457","FTY657","FTY1110","FTY485","FTY315","FTY1852","FTY1107","FTY489","FTY522","FTY654","FTY110","FTY881","DYS470","FTY534","FTY1101","FTY1933","FTY936","FTY772","FTY1047","FTY2351","FTY680","FTY2254","FTY1030","FTY2083","FTY287","FTY797","FTY2366","FTY951","FTY774","FTY1116","DYS621","FTY885","FTY1012","DYF394u1","FTY888","FTY648","FTY725","FTY643","FTY927","FTY1022","DYS559","FTY933","FTY633","FTY900","FTY1848","FTY478","FTY533","FTY467","FTY1900","FTY421","DYS612u5","DYS595","FTY668","FTY1028","DYS579","FTY646","FTY1026","FTY801","FTY432","FTY2301","FTY466","FTY531","FTY1083","FTY997","FTY1114","FTY830","FTY655","FTY910","FTY1143","FTY572","DYS506","DYS558u2","FTY1127","FTY712","FTY634","FTY499","FTY1094","FTY724","FTY535","FTY552","FTY430","FTY588","DYS609","FTY961","FTY813","FTY656","FTY839","FTY882","FTY789","FTY459","FTY1037","FTY435","FTY1157","FTY1546","FTY2242","FTY720","DYS614u10","FTY887","FTY473","FTY816","FTY1060","FTY472","FTY943","FTY1482","FTY565","DYS629","FTY891","FTY438","DYS582","FTY468","FTY649","FTY606","FTY792","FTY1087","FTY895","DYS614u3","FTY574","FTY1556","FTY560","FTY694","FTY1097","DYS518u6","FTY876","FTY2443","FTY990","FTY934","FTY465","FTY1046","FTY1312","FTY751","FTY1915","FTY678","FTY419","FTY958","FTY456","DYS592u1","FTY1172","FTY416","FTY650","FTY808","FTY1542","DYS721u1","FTY746","FTY2233","FTY892","FTY845","FTY1034","FTY498","DYF382u1","FTY637","DYS588","FTY947","DYS543u3","FTY644","DYS703","DYS649","FTY212","FTY781","FTY1106","FTY592","FTY848","FTY940","FTY903","FTY1081","FTY952","DYS706u2","FTY1084","FTY700","FTY482","DYS625u7","FTY960","FTY586","FTY1330","FTY1119","FTY1762","FTY1010","FTY417","FTY716","DYS627u3","FTY1942","FTY1067","FTY841","FTY1031","FTY1082","FTY1023","FTY475","FTY222","FTY962","FTY423","FTY664","DYS517u1","FTY1076","FTY1557","FTY1154","DYS681","FTY1002","DYS706u1","FTY1174","FTY986","FTY651","FTY886","FTY1078","DYS546u4","FTY1092","FTY2471","FTY1111","FTY461","FTY323","FTY1013","FTY506","FTY550","FTY767","FTY721","DYS548","FTY559","DYS705","FTY640","FTY613","FTY402","FTY2426","FTY755"]
    mustr=mustr[0:111]
    diffs=np.array([-5,-4,-3,-2,-1,1,2,3,4,5])
    probs=np.array([0.0005,0.0005,0.002,0.016,0.481,0.481,0.016,0.002,0.0005,0.0005])
    gd=np.zeros(111,dtype=int)
    nmut=0
    for i in np.arange(gens):
        rand1=np.random.random(111)
        rand2=np.random.choice(diffs, 111, p=probs)
        g=np.where(rand1<mustr,1,0)*np.where(rand2<0.5,-1,1)
        gd+=g
        nmut+=np.sum(np.abs(g))
        if (np.sum(np.abs(g))>0):
            print ("    Generation:",i)
            for j in np.arange(111):
                if (g[j]!=0):
                    print ("        Mutation:",j,g[j],strnames[j])
    #print (gd)
    return gd,nmut

def snps(tmrca,coverage):
    prob=8.2e-10*coverage
    snp=0
    a=np.where(np.random.random(int(tmrca))<prob,1,0)
    return np.sum(a)

# ====================================
# Functions defining TMRCA calculation
# ====================================

# Generate weighting factors for a specific genetic distance (g), number of mutations (m) and multi-step combination (Q)
# Calculates the three-line equation for \varpi_g,m[Q] in the accompanying LaTeX document
# Each factors[] represents one of the factors in the equation
# countq = number of multi-step mutations in Q+ (latterly Q-) which have multi-step degree i
def gdq_varpi(g,m,wp,wn,omegap,omegan,qp,qn,kp,kn,epsp,epsn,nqp,nqn,maxnq):
	factors=np.ones(8,dtype=float)
	factors[0]=scipy.special.comb(m,kp,exact=True) # First line - probability of getting kp positives and kn negatives
	factors[1]=wp**kp*wn**kn
	for i in range(maxq): # This performs the loop over multi-step size n
		countq=len(qp[np.where(qp==i)]) # Second line - product of all the probabilities of getting Q multi-step mutations from kp positives
		if (countq>0):
			factors[2]*=scipy.special.comb(kp,countq,exact=True)
			factors[3]*=(omegap[i]/wp)**countq
			factors[4]*=(1-omegap[i]/wp)**(kp-countq)
			#print ("i,countq,kp:",i,countq,kp,scipy.special.comb(kp,countq))
		countq=len(qn[np.where(qn==i)]) # Third line - and probability of getting Q multi-step mutations from kn negatives
		if (countq>0):
			factors[5]*=scipy.special.comb(kn,countq,exact=True)
			factors[6]*=(omegan[i]/wn)**countq
			factors[7]*=(1-omegan[i]/wn)**(kn-countq)
			#print ("i,countq,kn:",i,countq,kn,scipy.special.comb(kn,countq))
	varpi=np.prod(factors) # Perform final multiplication
	#if (varpi>1.e-3):
	#	trace(10, 'g={} m={} q={},{} eps={},{} k={},{} nq={},{} varpi={}'.format(g,m,qp,qn,epsp,epsn,kp,kn,nqp,nqn,varpi))
	return varpi

# Generate weighting factors for obtaining a genetic distance g, given m mutations
# (\varpi_g,m in accompanying LaTeX document)
def gd_weight(g,m,wp,wn,omegap,omegan,maxq,maxnq):
	if (g==0 and m==0): # Special case if no mutations
		varpi=1.
	else: # If some mutations
		varpi=0.
		# General a set of possible multi-step mutations and loop over those sets
		qp0=np.ones(m,dtype=int) # Set up lists of positive/negative mutations
		qn0=np.ones(m,dtype=int)
		qsets=list(itertools.combinations_with_replacement(range(1,maxq),min(maxnq,m)))
		for qpset in qsets:
			for qnset in qsets:
				qp=qp0 # Get original sets of Q+ and Q-
				qn=qn0
				multip=np.array(list(reversed(qpset))) # Generate lists of Q+ and Q-
				multin=np.array(list(reversed(qnset)))
				qp[:len(multip)]=multip
				qn[:len(multip)]=multin
				nqp=len(qp[np.where(qp>1)]) # Count number of multistep mutations (positive & negative)
				nqn=len(qn[np.where(qn>1)])
				if (nqp+nqn<=maxnq): # Only use those where combined Q+ and Q- < maximum no. allowed multi-step mutations
					epsp=np.sum(qp[np.where(qp>1)])-nqp  # Calculate additional repeats caused by multi-step mutations (\varepsilon)
					epsn=np.sum(qn[np.where(qn>1)])-nqn
					kp=(m+g-epsp+epsn)/2. # Calculate number of required positive and negative mutations (k_+, k_-)
					kn=(m-g+epsp-epsn)/2.
					# If an positive and integer number of +/- mutations, if the required genetic distance is obtained, if the number of multistep markers is less than the total
					if (kp>=0 and kn>=0 and kp==int(kp) and kn==int(kn) and kp+kn==m and kp+epsp-kn-epsn==g and nqp<=kp and nqn<=kn):
						qp=qp[:int(kp)]
						qn=qn[:int(kn)]
						varpi+=gdq_varpi(g,m,wp,wn,omegap,omegan,qp,qn,int(kp),int(kn),int(epsp),int(epsn),nqp,nqn,maxnq) # Calculate probability weight
	return varpi

# This returns the probability density functions for a generic Y-STR of genetic distance (g) in mutation timescales
# This calculates p_g(\bar{m}_s) in the accompanying LaTeX document
# ***Note: still need to code in the case where g is negative
# ***where wp==wn, gmweightp = gmweightn
# ***where wp!=wn a new computation is needed
def generate_genpdfstr(mtimes,pdf_str_m,maxg,maxm,maxq,maxnq,maxnm,mres,omegap,omegan):
	gmweight=np.zeros((maxg,maxm)) # Weight generation
	wp=np.sum(omegap[1:]) # Calculate w+ and w-, the weight of obtaining a positive/negative result overall
	wn=np.sum(omegan[1:])
	# Loop over genetic distances and numbers of mutations
	for g in range(maxg):
		for m in range(maxm):
			gmweight[g,m]=gd_weight(g,m,wp,wn,omegap,omegan,maxq,maxnq) # Get weights
			if (gmweight[g,m]>0): # If weight is non-zero
				#trace (3, 'weight[{},{}]={}'.format(g,m,gmweight[g,m]))
				for i in range(len(mtimes)): # Add Poisson distribution of mean m * appropriate weight to overall PDF
					#print(int(m),mtimes[i],scipy.stats.poisson.pmf(int(m),mtimes[i]))
					pdf_str_m[g,i]+=gmweight[g,m]*scipy.stats.poisson.pmf(int(m),mtimes[i])
	#Outputs first few results for inspection
	#np.savetxt("str_pdf.dat", np.c_[mtimes,pdf_str_m[0,:],pdf_str_m[1,:],pdf_str_m[2,:],pdf_str_m[3,:]], delimiter=' ')

# This calculates the probability mass functions for speicifc Y-STRs of genetic distance (g) in physical timescales (generations)
# This calculates p_g(t) in the accompanying LaTeX document
def generate_pdfstr(mtimes,times,pdf_str_m,pdf_str_t,mustr,mustrunc,maxtime,maxg):
	for i in range(len(mustr)):
		# Generate distribution in mu
		#muscale=np.arange(0,maxtime,0.5*mustr[i])
		#mu=np.zeros(len(muscale))
		
		# Need to compute time [t] = mutations [m] * mutation rate [mu]
		# Both m and mu are PDFs, to give t as a PDF
		# Seemingly no functional way of computing the product of two PDFs in Python
		# So let's do this ourselves
		# Not doing this as a function because the implementation for SNPs is subtly different
		# to reduce computational timeframe here
		# Let's represent mu by a log-normal distribution
		logmu=log10(1/mustr[i])
		sigma=log10(1/(1-mustrunc[i]/mustr[i]))
		#trace (5, 'STR {} of {}: mu={} yr unc={}% logmean={} logsigma={}'.format(i+1,len(mustr),1/mustr[i],mustrunc[i]/mustr[i]*100,logmu,sigma))
		# Sample points from this log-normal distribution
		ppfsamp=np.arange(0.02,0.98,0.04) # This samples 24 points - more may be required if fractional uncertainty is large
		ppfsamps=len(ppfsamp)
		lnsamp=scipy.stats.lognorm.ppf(ppfsamp,sigma,logmu-1)
		# Then translate m to t using that sampled mu
		for musamp in lnsamp:
			xtimes=mtimes*10**musamp
			for g in range(maxg):
				pdf_str_t[i,g,:]+=np.interp(times,xtimes,pdf_str_m[g,:])/ppfsamps
				#print (g,10**musamp,xtimes,pdf_str_t[i,g,:])
		#Outputs first few results for inspection
		#np.savetxt("str_pdf_t.dat", np.c_[times,pdf_str_t[0,0,:],pdf_str_t[0,1,:],pdf_str_t[0,2,:],pdf_str_t[0,3,:]], delimiter=' ')
				

# STR TMRCA calculator for one test
def strage(times,pdf_str_t,ancht,derht):
	nstr=min(len(ancht),len(derht)) # Total number of STRs
	gd=abs(ancht-derht) # Find GD on each STR *** Remove abs to allow negative genetic distances if wp!=wn
	#trace (5, 'GD:{}'.format(gd))
	tmrca_str=np.ones(len(times)) # Setup TMRCA array
	strcomp=0
	for i in range(nstr): # Loop over STRs
		if (ancht[i]>0 and derht[i]>0): # If non-null
			strcomp+=1
			tmrca_str*=pdf_str_t[i,gd[i],:] # *** Needs edit to allow negative GD for wp!=wn
	# Normalise
	psum=sum(tmrca_str)
	tmrca_str/=psum
	#trace(5, '{} STRs compared'.format(strcomp))
	#Output result for inspection
	#np.savetxt("tmrca_str.dat", np.c_[times,tmrca_str], delimiter=' ')
	return tmrca_str
			

# SNP TMRCA calculator for one test
def snptmconv(times,stimes,tsnpm,testcombcov,musnp,musnpunc):
	pdf_snp=np.zeros(len(times))
	# As elsewhere, we need to compute time [t] = mutations [m] * mutation rate [mu]
	# Both m and mu are PDFs, to give t as a PDF
	# Let's represent mu by a log-normal distribution
	logmu=log10(1/testcombcov/musnp)
	sigma=log10(1/(1-musnpunc/musnp))
	#trace (5, 'SNPs: cov={} mu={} cov*mu={} yr/SNP unc={}% logmean={} logsigma={}'.format(testcombcov,musnp,1/(testcombcov*musnp),musnpunc/musnp*100,logmu,sigma))
	# Sample points from this log-normal distribution
	ppfsamp=np.arange(0.02,0.98,0.04) # This samples 24 points - more may be required if fractional uncertainty is large
	ppfsamps=len(ppfsamp)
	lnsamp=scipy.stats.lognorm.ppf(ppfsamp,sigma,logmu-1)
	# Then translate m to t using that sampled mu
	for musamp in lnsamp:
		xtimes=stimes*10**musamp
		pdf_snp+=np.interp(times,xtimes,tsnpm)/ppfsamps
		#print (10**musamp,xtimes,pdf_snp)
	#Outputs first few results for inspection
	#np.savetxt("snp_pdf_t.dat", np.c_[times,pdf_snp,], delimiter=' ')
	return pdf_snp
	
# Paper trail estimator
def paperpdf(dates,pdftype,t,s=1,a=1,psi=0):
	pdf=np.ones(len(dates))
	if (pdftype=="delta"):
		pdf=psi
		idx=(np.abs(dates-t)).argmin()
		pdf[idx]=1
	elif (pdftype=="smooth"):
		pdf=scipy.stats.norm.pdf((t-dates)/s)*(1-psi)+psi
	elif (pdftype=="step-up"):
		pdf[dates>t]=psi
	elif (pdftype=="step-down"):
		pdf[dates<t]=psi
	elif (pdftype=="smooth-start"):
		pdf=scipy.stats.norm.cdf((t-dates)/s)*(1-psi)+psi
	elif (pdftype=="smooth-end"):
		pdf=scipy.stats.norm.cdf((dates-t)/s)*(1-psi)+psi
	elif (pdftype=="ln-start"):
		pdf=scipy.stats.lognorm.cdf((t-dates)/a,log10(s))*(1-psi)+psi
	elif (pdftype=="ln-end"):
		pdf=scipy.stats.lognorm.cdf((dates-t)/a,log10(s))*(1-psi)+psi
	else:
		# May ultimately want to interpret this as a filename
		print ("Paper trail PDF type not recognised:{}".format(pdftype))
	return pdf
	
# Ages to dates calculator
def ages2dates(pdf_times,dates,times,timestep,nowtime,t0,st):
	# As elsewhere, we need to compute dates [dates] = zero [t0] - times [t]
	# Both m and mu are PDFs, to give t as a PDF
	# Let's represent zero by a normal distribution
	pdf_zero=scipy.stats.norm.pdf((times-(nowtime-t0))/st)
	# Convolve times with zero to get dates
	pdf_dates=scipy.convolve(pdf_times,pdf_zero,mode='full')
	pdf_dates=pdf_dates[:len(dates)]
	#np.savetxt("ages2dates.dat", np.c_[dates,pdf_dates,times,pdf_times,pdf_zero], delimiter=' ')
	return pdf_dates
	
def test2(ht0,ht1,ht2,testsnpdist1,testsnpdist2,testcombcov,testpaper1,testpaperunc1,testpaperalpha1,testpapertype1,testpaper2,testpaperunc2,testpaperalpha2,testpapertype2,tystpaperuncs,testpapertypes):
	# Test data

	# Choose what tests to include
	do_strs=1
	do_snps=1
	do_indels=0
	do_paper=0
		
	if (do_paper>0):
		tmrca_paper=np.ones(len(dates))
		# TMRCA from paper results from the conjunction of TMRCA 1 & 2,
		# so it is the maximum probability of being before this date.
		# This allows for one line to join the other before the oldest
		# most-distant known ancestor.
		tmrca_paper*=np.maximum(paperpdf(dates,testpapertype1,testpaper1,testpaperunc1,testpaperalpha1),paperpdf(dates,testpapertype2,testpaper2,testpaperunc1,testpaperalpha1))
		# Account for shared surname
		tmrca_paper*=paperpdf(dates,testpapertypes,testpapers,testpaperuncs)
		#np.savetxt("tpaper.dat", np.c_[dates,tmrca_paper], delimiter=' ')
	else:
		tmrca_paper=np.ones(len(dates))
	
	if (do_strs>0):
		# Calculate TMRCA from STRs
		#trace(0, 'Calculating STR-based TMRCA')
		tstr1=strage(times,pdf_str_t,ht0,ht1)
		#np.savetxt("tstr1.dat", np.c_[times,tstr1], delimiter=' ')
		tstr2=strage(times,pdf_str_t,ht0,ht2)
		#np.savetxt("tstr2.dat", np.c_[times,tstr2], delimiter=' ')
		# Combine these TMRCAs
		tmrca_str=tstr1*tstr2
		# Normalise PDF
		psum=sum(tmrca_str)
		tmrca_str/=psum
		np.savetxt("tstr.dat", np.c_[times,tmrca_str], delimiter=' ')
		# Get confidence intervals
		cumtmrca=np.cumsum(tmrca_str)
		#print ('STR Central estimate:          {} years'.format(times[len(cumtmrca[cumtmrca<0.5])]))
		#print ('STR 68.3% confidence interval: {} -- {} years'.format(times[len(cumtmrca[cumtmrca<0.1585])],times[len(cumtmrca[cumtmrca<0.8415])]))
		#print ('STR 95.0% confidence interval: {} -- {} years'.format(times[len(cumtmrca[cumtmrca<0.025])],times[len(cumtmrca[cumtmrca<0.975])]))
		#print ('STR 99.5% confidence interval: {} -- {} years'.format(times[len(cumtmrca[cumtmrca<0.0025])],times[len(cumtmrca[cumtmrca<0.9975])]))
	else:
		tmrca_str=np.ones(len(times))
		tstr1=np.ones(len(times))
		tstr2=np.ones(len(times))

	if (do_snps>0):
		#trace(0, 'Calculating SNP-based TMRCA')
		# Calculate TMRCA from SNPs
		stimes=np.arange(0,maxsnps,snpstep)
		tsnpm1=scipy.stats.poisson.pmf(testsnpdist1,stimes)
		tsnpm2=scipy.stats.poisson.pmf(testsnpdist2,stimes)
		tsnpm=tsnpm1*tsnpm2
		# Convert to physical timescale
		tmrca_snp=snptmconv(times,stimes,tsnpm,testcombcov,musnp,musnpunc)
		tsnp1=snptmconv(times,stimes,tsnpm1,testcombcov,musnp,musnpunc)
		#np.savetxt("tsnp1.dat", np.c_[times,tsnp1], delimiter=' ')
		tsnp2=snptmconv(times,stimes,tsnpm2,testcombcov,musnp,musnpunc)
		#np.savetxt("tsnp2.dat", np.c_[times,tsnp2], delimiter=' ')
		# Normalise PDF
		psum=sum(tmrca_snp)
		tmrca_snp/=psum
		np.savetxt("tsnp.dat", np.c_[times,tmrca_snp], delimiter=' ')
		# Get confidence intervals
		cumtmrca=np.cumsum(tmrca_snp)
		#print ('SNP Central estimate:          {} years'.format(times[len(cumtmrca[cumtmrca<0.5])]))
		#print ('SNP 68.3% confidence interval: {} -- {} years'.format(times[len(cumtmrca[cumtmrca<0.1585])],times[len(cumtmrca[cumtmrca<0.8415])]))
		#print ('SNP 95.0% confidence interval: {} -- {} years'.format(times[len(cumtmrca[cumtmrca<0.025])],times[len(cumtmrca[cumtmrca<0.975])]))
		#print ('SNP 99.5% confidence interval: {} -- {} years'.format(times[len(cumtmrca[cumtmrca<0.0025])],times[len(cumtmrca[cumtmrca<0.9975])]))
	else:
		tmrca_snp=np.ones(len(times))
		tsnp1=np.ones(len(times))
		tsnp2=np.ones(len(times))
	
	if (do_indels>0):
		trace(0, 'Calculating indel-based TMRCA')
		# Calculate TMRCA from indels
		stimes=np.arange(0,maxsnps,snpstep)
		tsnpm1=scipy.stats.poisson.pmf(testindeldist1,stimes)
		tsnpm2=scipy.stats.poisson.pmf(testindeldist2,stimes)
		tsnpm=tsnpm1*tsnpm2
		# Convert to physical timescale
		tmrca_indel=snptmconv(times,stimes,tsnpm,testcombcov,muindel,muindelunc)
		tindel1=snptmconv(times,stimes,tsnpm,testcombcov,musnp,musnpunc)
		#np.savetxt("tindel1.dat", np.c_[times,tindel1], delimiter=' ')
		tindel2=snptmconv(times,stimes,tsnpm,testcombcov,musnp,musnpunc)
		#np.savetxt("tindel2.dat", np.c_[times,tindel2], delimiter=' ')
		# Normalise PDF
		psum=sum(tmrca_indel)
		tmrca_indel/=psum
		np.savetxt("tindel.dat", np.c_[times,tmrca_indel], delimiter=' ')
		# Get confidence intervals
		cumtmrca=np.cumsum(tmrca_indel)
		#print ('Indel Central estimate:          {} years'.format(times[len(cumtmrca[cumtmrca<0.5])]))
		#print ('Indel 68.3% confidence interval: {} -- {} years'.format(times[len(cumtmrca[cumtmrca<0.1585])],times[len(cumtmrca[cumtmrca<0.8415])]))
		#print ('Indel 95.0% confidence interval: {} -- {} years'.format(times[len(cumtmrca[cumtmrca<0.025])],times[len(cumtmrca[cumtmrca<0.975])]))
		#print ('Indel 99.5% confidence interval: {} -- {} years'.format(times[len(cumtmrca[cumtmrca<0.0025])],times[len(cumtmrca[cumtmrca<0.9975])]))
	else:
		tmrca_indel=np.ones(len(times))
		tindel1=np.ones(len(times))
		tindel2=np.ones(len(times))

	tmrca=tmrca_str*tmrca_snp*tmrca_indel
	tmrca1=tstr1*tsnp1*tindel1
	tmrca2=tstr2*tsnp2*tindel2
	# Normalise PDF
	tmrca/=np.sum(tmrca)
	tmrca1/=np.sum(tmrca1)
	tmrca2/=np.sum(tmrca2)
	#np.savetxt("tmrca.dat", np.c_[times,tmrca], delimiter=' ')
	# Get confidence intervals
	cumtmrca=np.cumsum(tmrca)
	#print (cumtmrca)
	#print ('Combined Central estimate:          {} years'.format(times[len(cumtmrca[cumtmrca<0.5])]))
	#print ('Combined 68.3% confidence interval: {} -- {} years'.format(times[len(cumtmrca[cumtmrca<0.1585])],times[len(cumtmrca[cumtmrca<0.8415])]))
	#print ('Combined 95.0% confidence interval: {} -- {} years'.format(times[len(cumtmrca[cumtmrca<0.025])],times[len(cumtmrca[cumtmrca<0.975])]))
	#print ('Combined 99.5% confidence interval: {} -- {} years'.format(times[len(cumtmrca[cumtmrca<0.0025])],times[len(cumtmrca[cumtmrca<0.9975])]))
	# Convert to dates
	dt=dt0/sqrt(2) # Average birth date uncertainty is reduced by factor sqrt(2) for two testers
	st=sqrt(st0**2+dt**2) # Add uncertainty in the mean
	dmrca=ages2dates(tmrca,dates,times,timestep,nowtime,t0,st) # Convert TMRCA to dates
	#print (len(dmrca),len(tmrca_paper))
	dmrca*=tmrca_paper # Add in paper-trail genealogy limits
	# Normalise PDF
	psum=sum(dmrca)
	dmrca/=psum
	#np.savetxt("dmrca.dat", np.c_[dates,dmrca], delimiter=' ')
	# Get confidence intervals
	cumdmrca=np.cumsum(dmrca)
	#print (cumdmrca)
	#centiles=[0.5,0.1585,0.8415,0.025,0.975,0.0025,0.9975]
	#estimators=dates[len(cumdmrca[cumdmrca<centiles])]
	#ad=(dates>0)
	#print ('Combined Central estimate:          {} CE'.format(dates[len(cumdmrca[cumdmrca<0.5])]))
	#print ('Combined 68.3% confidence interval: {} -- {} CE'.format(dates[len(cumdmrca[cumdmrca<0.8415])],dates[len(cumdmrca[cumdmrca<0.1585])]))
	#print ('Combined 95.0% confidence interval: {} -- {} CE'.format(dates[len(cumdmrca[cumdmrca<0.975])],dates[len(cumdmrca[cumdmrca<0.025])]))
	#print ('Combined 99.5% confidence interval: {} -- {} CE'.format(dates[len(cumdmrca[cumdmrca<0.9975])],dates[len(cumdmrca[cumdmrca<0.0025])]))

	return tmrca,tmrca1,tmrca2


# ==========================================
# Main code to run the examples in the paper
# ==========================================
# Set global parameters
# ---------------------

# Zero points
t0 = 1950.0
st0 = 1.0 # Uncertainty in the mean
dt0 = 10.0 # Standard deviation
# STR rates and uncertainties
mustr=np.array([0.001776,0.003558,0.002262963333333,0.00214183,0.001832,0.003458,0.0002157006,0.000537249333333,0.003600416666667,0.002361616666667,0.000516113666667,0.0026555,0.007048133333333,0.000523,0.001525,0.000868596333333,0.000918865,0.0019482,0.001313576666667,0.001448,0.0089895,0.001874,0.0032,0.004241,0.003716,0.002626356666667,0.0022795,0.000496,0.001141,0.004139643333333,0.002660216666667,0.0096684,0.006975733333333,0.014357,0.018449,0.002620776666667,0.000612560666667,0.001436,0.000990948666667,0.000433,0.000319,0.000254032,0.00166647,0.000919199333333,0.0002167,0.00282426,0.0017663,0.000236,0.00229,0.001527,0.0034075,0.000467199,0.000203582333333,0.0003603,0.005522,0.000264693666667,0.0027805,0.005328563333333,0.0020715,0.0028765,0.00055675,0.001144006333333,0.00081927,0.001894,0.000297,0.000300432666667,0.001237244666667,0.018279,0.00106261,6.96637E-05,0.000826361,0.001597966666667,0.007726,0.001002,0.00064516,0.00218298,0.001634593333333,0.004000833333333,0.0007529,0.002278673333333,0.000260055333333,0.0027318,0.001357,0.000812522666667,0.0013825,0.001339923333333,0.00105415,0.001369627,0.00373403,0.00076,0.002423076666667,0.00084,0.00178172,0.016378,0.000278238,0.007583,0.004142,0.00387799,0.006949,0.0024505,0.00176344,0.00301,0.0002895845,0.003112,0.0008633,0.00133475,0.0007121,0.0026829,0.001059957666667,0.002533206666667,0.000909901666667,1.00E-06,1.00E-06,1.00E-06,1.00E-06,1.00E-06,1.00E-06,1.00E-06,0.000866329393708,0.000646939655185,1.00E-06,1.00E-06,1.00E-06,1.00E-06,0.000185879225554,1.00E-06,1.00E-06,1.00E-06,7.18628072330345E-06,1.00E-06,1.00E-06,1.00E-06,1.00E-06,1.00E-06,1.00E-06,1.00E-06,1.00E-06,1.00E-06,1.00E-06,1.00E-06,1.00E-06,4.30389844826262E-05,1.00E-06,1.00E-06,1.00E-06,1.00E-06,1.00E-06,1.00E-06,1.00E-06,1.00E-06,4.31170729064071E-05,1.00E-06,1.00E-06,1.00E-06,1.00E-06,1.00E-06,1.00E-06,1.00E-06,1.00E-06,5.77076443913719E-05,7.19120620701156E-06,1.00E-06,1.00E-06,1.00E-06,2.88041462217885E-05,1.00E-06,7.17644996995205E-06,1.00E-06,1.00E-06,1.00E-06,1.00E-06,7.20107745785662E-06,7.19613844723963E-06,1.00E-06,1.00E-06,1.00E-06,0.000230751008629,1.00E-06,1.43627104792525E-05,6.48782185068035E-05,7.22090148387681E-06,1.00E-06,6.50328710086277E-05,8.6689637041137E-05,1.00E-06,0.001323535604723,1.00E-06,1.00E-06,1.00E-06,1.00E-06,2.16477441055493E-05,7.20602325276963E-06,1.00E-06,1.00E-06,1.43823988732946E-05,1.00E-06,1.00E-06,0.008133689075989,1.00E-06,1.00E-06,1.00E-06,1.00E-06,1.00E-06,1.00E-06,1.00E-06,1.00E-06,1.00E-06,1.44120328808033E-05,0.000178671921343,1.00E-06,1.44716687654042E-05,1.00E-06,7.20107745785662E-06,8.62299433967952E-05,4.33247885674273E-05,1.00E-06,1.00E-06,2.90634603345777E-05,1.00E-06,0.003577001096869,1.00E-06,1.00E-06,1.00E-06,1.00E-06,1.00E-06,7.230854483894E-06,7.21593525175012E-06,1.00E-06,1.00E-06,7.34217624636386E-06,1.00E-06,1.00E-06,7.32679459221162E-06,7.26590710251738E-06,1.00E-06,1.00E-06,1.00E-06,1.00E-06,0.000340653719367,1.00E-06,7.49426418290361E-06,1.00E-06,8.71104600989299E-05,7.23584128005022E-06,7.23085448384135E-06,1.00E-06,7.26087879312772E-06,1.00E-06,1.00E-06,1.00E-06,1.00E-06,0.000107848004839,1.00E-06,0.00070976761243,0.002858815116904,1.00E-06,2.16031711042177E-05,1.00E-06,7.25084302421095E-06,7.311477251621E-06,0.001408707269388,1.00E-06,1.00E-06,0.001773519683037,1.00E-06,0.000115613331671,1.00E-06,1.00E-06,0.000156930073995,1.00E-06,1.00E-06,1.00E-06,2.16180083626181E-05,1.00E-06,2.16626426743719E-05,7.21097584608153E-06,0.000264790198753,6.55725075769289E-05,1.4501672167711E-05,0.000957091249245,1.00E-06,1.00E-06,7.23584128005022E-06,0.004884093197363,5.80669449264951E-05,7.23085448384135E-06,0.000375542091327,9.32869226725564E-05,7.39913247958075E-06,1.00E-06,2.89234179353917E-05,1.00E-06,1.45822928440098E-05,0.000240161874603,1.00E-06,1.45318142050708E-05,2.2098698583111E-05,0.00054974441262,1.00E-06,1.00E-06,1.00E-06,0.000814739188919,0.001692263028863,7.29115347885296E-06,1.00E-06,0.00029993372441,0.000151842964233,1.00E-06,3.04115068291627E-05,1.00E-06,0.002179366480775,1.00E-06,0.001176200984885,1.00E-06,3.62038277773263E-05,0.000228878374873,1.00E-06,1.00E-06,7.28609017780372E-06,0.000695661057329,0.002038775146703,0.00034583869351,2.17325060312092E-05,0.000202042136976,2.90837695247712E-05,0.000224370527201,1.00E-06,2.91039385743883E-05,1.00E-06,7.26892826973318E-05,1.00E-06,7.2962238220587E-06,1.00E-06,0.000479061056326,1.00E-06,7.21097584608153E-06,1.00E-06,0.000345084936576,1.00E-06,6.48194321241704E-05,1.00E-06,1.00E-06,1.00E-06,1.00E-06,4.3275773887849E-05,1.00E-06,1.00E-06,1.00E-06,1.00E-06,1.00E-06,1.00E-06,2.22078408156419E-05,1.00E-06,1.00E-06,7.311477251621E-06,0.000137443809509,1.00E-06,1.00E-06,7.62497809308834E-06,0.000228722497731,2.17224425871507E-05,1.4501672167711E-05,2.31383827594682E-05,2.17025370439257E-05,1.45924335008614E-05,1.00E-06,2.2514887404072E-05,1.00E-06,0.000727471884019,0.004627006466369,0.000908822480141,1.00E-06,1.00E-06,1.00E-06,6.13784627205385E-05,1.00E-06,1.00E-06,0.000253104032158,1.00E-06,1.00E-06,1.00E-06,0.005655396000866,1.00E-06,0.002290119886728,1.00E-06,1.00E-06,0.000141953388767,1.00E-06,0.002976316037582,0.000794431508604,7.32168168604791E-06,1.00E-06,0.000120857824116,1.00E-06,0.001113326336591,1.00E-06,1.00E-06,0.000152916056261,1.00E-06,0.005242230390096,1.00E-06,2.34661382616262E-05,7.26590710251738E-06,2.23708698150525E-05,1.00E-06,4.73440664708861E-05,1.00E-06,1.00E-06,1.00E-06,1.00E-06,1.00E-06,0.000327325431358,0.000122996751763,1.00E-06,1.00E-06,8.73612992481482E-05,0.00153550001202,2.21921831100369E-05,7.24083495932796E-06,0.000376992758557,1.00E-06,1.00E-06,2.16577382194611E-05,8.10249390867033E-05,0.008714410639185,0.000829703192635,0.000872147285301,7.60839003340416E-06,1.00E-06,0.004355995076024,2.28914976825357E-05,0.001977045000582,0.002592997087187,1.00E-06,1.00E-06,2.21609338427711E-05,1.52057374107413E-05,0.004876561421801,0.001447751657171,0.001501338426772,7.43583972788376E-06,0.000896489821155,1.00E-06,1.00E-06,7.19613844723963E-06,1.00E-06,1.00E-06,1.00E-06,3.62842807279286E-05,1.00E-06,1.00E-06,1.00E-06,7.3887111662081E-06,1.00E-06,0.000125529750045,1.00E-06,1.00E-06,0.000129422649816,0.001287373752605,0.001351076257623,1.00E-06,3.63040440504395E-05,7.32679459221162E-06,0.000123344245185,7.22476883737539E-05,1.00E-06,7.39391815085335E-06,7.87685424626355E-06,0.000159909999226,1.00E-06,7.38351151023684E-06,1.00E-06,3.20853084707844E-05,0.000701370982279,7.75788957522413E-05,7.33704185737334E-06,7.4782393842272E-06,7.40435416800369E-06,0.000994525963611,1.00E-06,7.06549683294134E-05,1.00E-06,0.002030114945391,1.53728496059504E-05,2.99129575368543E-05,0.000473332555761,1.00E-06,0.001022672298961,0.000323892700767,0.00051820765804,0.002086720643268,2.91443607112809E-05,1.00E-06,0.000195652693676,7.69770348940673E-06,1.00E-06,0.000651357389689,0.001469025770727,1.00E-06,5.36365594267711E-05,1.00E-06,0.001009358042687,1.00E-06,0.00093988518356,2.30761053423314E-05,0.002999455874981,1.00E-06,0.001095112066371,1.00E-06,1.00E-06,3.55958566255592E-05,1.00E-06,1.00E-06,1.00E-06,1.00E-06,0.000188971689956,0.001242269874803,1.60182558783611E-05,1.00E-06,4.78348932246706E-05,0.00038676857562,1.00E-06,2.59273318752663E-05,1.53391372164636E-05,7.54275331133574E-06,0.006006121688146,1.00E-06,1.00E-06,1.00E-06,0.002345691789981,1.00E-06,1.00E-06,0.002382318331383,1.00E-06,7.3700230952696E-05,0.000391849483953,1.00E-06,0.001089617317705,0.000111346867817,0.000802564339049,0.000153930897162,1.00E-06,0.000107485260618,1.00E-06,1.00E-06,1.52167800668037E-05,0.000721185331166,6.08004789120523E-05,0.000561307787544,2.25903683039772E-05,1.50207156135449E-05,1.00E-06,0.001393079169983,0.001079690824854,1.00E-06,5.4662827803081E-05,0.000910753943082,1.00E-06,0.00068480347357,0.001227366055884,0.000263443029256])
mustr=mustr[0:111]
mustrunc=mustr*0.5 # *** Mutation rate uncertainties - could replace with realistic uncertainties
mustrunc+=5.e-7
ygen=33 # Years per generation
ygenunc=2 # Years per generation uncertainty
mustr/=ygen # Convert mutation rates to years
mustrunc/=ygen*(1+ygenunc/ygen)
nstr=len(mustr) # Get maximum number of STRs to consider
# Mutation direction probabilities (positive/negative change in STR allele)
wp=0.5 # Equal probability of positive and negative mutations
wn=wp  # *** wn!=wp not coded here!
# STR multi-step probabilities
omegap=np.array([1,0.9621685978,0.032,0.004,0.0012649111,0.0004,0.000124649111,0.00004]) # Estimated from Ballantyne et al. (2010)
omegap/=2
omegan=omegap
# SNP mutation rate and uncertainty
musnp=8.330e-10 #7.547e-10 # SNP creation rate /base pair/year
musnpunc=0.800e-10 #0.661e-10
muindel=5.75e-11 # Indel rate (estimated from 315 indels versus 4137 SNPs in Build 37 report)
muindelunc=0.504e-11 # Based on quadrature sum of sqrt(315)/4137 and musnpunc/musnp
maxsnps=100 # Maximum number of mutations to consider when forming tree
snpstep=0.01 # Step size in mutation timescales
ylength=60000000 # Maximum number of base pairs to consider

# Generate timeline
maxtime=8000 # years into the past
timestep=1 #years
times=np.arange(0,maxtime,timestep)
nowtime=2020 # present day (AD/CE)
dates=np.arange(nowtime,nowtime-maxtime,-timestep)

# Set up parameters for generating the PDFs for STRs
maxg=8 # Max GD considered < 5 = quick, but probably normally want ~10 for a full tree
maxm=8 # Max number of mutations considered < close relationships ~5, probably normally want ~20 for a large haplogroup
maxnm=20 # Max number of mutation timescales to compute maxm mutations to < probably 30 if CDY fast
maxq=len(omegap) # Max GD caused by a multi-step mutation
maxnq=3 # Max number of multi-step mutations to consider in one lineage < 3 maybe ok but might want ~4
mres=0.01 # Resolution of initial probability calculation (mutation timescales) < may want finer timescale, maybe ok
mtimes=np.arange(0,maxnm,mres)
pdf_str_m=np.zeros((maxg,len(mtimes))) # Probability density function of obtaining genetic distance (g) on any given STR in mutation timescale (m)
pdf_str_t=np.zeros((len(mustr),maxg,len(times))) # Probability density function of obtaining genetic distance (g) on specific STRs in physical timescale (t)


# Random haplotype to start from, ensures STR alleles are positive
ht0=np.array([13,25,14,11,11,13,12,12,12,13,14,29,16,9,11,11,11,25,15,18,30,15,16,16,17,11,11,19,23,17,16,19,17,38,39,12,12,11,9,15,16,8,10,10,8,10,10,12,21,23,16,10,12,12,16,8,13,25,20,13,12,11,13,11,11,12,12,35,15,9,16,12,24,26,20,12,11,12,12,11,9,13,12,10,11,11,31,12,13,24,13,10,10,22,15,20,13,24,17,14,15,24,12,23,18,10,14,17,9,12,11])


# Generate the generic PDF in terms of mutation timescales (p_g(\bar{m}_s))
generate_genpdfstr(mtimes,pdf_str_m,maxg,maxm,maxq,maxnq,maxnm,mres,omegap,omegan)
# Generate the specific PDFs for each STR
generate_pdfstr(mtimes,times,pdf_str_m,pdf_str_t,mustr,mustrunc,maxtime,maxg)





# Randomly seed the simulation
np.random.seed(1)



# Run examples
# ------------

# Example 1
print ("--- EXAMPLE 1 ---")
testpaper1=1950 # Paper trail info for line 1
testpaperunc1=15
testpaperalpha1=25
testpapertype1="smooth-start"
testpaper2=1950 # Paper trail info for line 2
testpaperunc2=15
testpaperalpha2=25
testpapertype2="smooth-start"
testpapers=0 # Surname restriction
testpaperuncs=1
testpapertypes="smooth-end" # -end for shared surname, -start for not shared

coverage=int(np.random.normal(loc=14.e6,scale=1.e6,size=1))
print ("Coverage=",coverage)

print ("Sample D to A")
tmrca=150
gens=generations(tmrca)
print ("Generations=",gens)
gd_da,nmut=strs(gens)
print ("GD=",np.sum(np.abs(gd_da)),"N_mut=",nmut)
snp_da=snps(tmrca,coverage)
print ("SNPs=",snp_da)
print ("")

print ("Sample D to B")
tmrca=150
gens=generations(tmrca)
print ("Generations=",gens)
gd_db,nmut=strs(gens)
print ("GD=",np.sum(np.abs(gd_db)),"N_mut=",nmut)
snp_db=snps(tmrca,coverage)
print ("SNPs=",snp_db)
print ("")

# Compute TMRCA D
ht1=ht0+gd_da
ht2=ht0+gd_db
hta=np.ndarray.flatten(scipy.stats.mode(np.array((ht0,ht1,ht2))).mode)
print ("Ex1D",hta)
print ("Ex1A",ht1)
print ("Ex1B",ht2)
output_tmrca_d,tmrca1,tmrca2=test2(hta,ht1,ht2,snp_da,snp_db,coverage,testpaper1,testpaperunc1,testpaperalpha1,testpapertype1,testpaper2,testpaperunc2,testpaperalpha2,testpapertype2,testpaperuncs,testpapertypes)
cumtmrca=np.cumsum(output_tmrca_d)
print ('Combined Central estimate:          {} years'.format(times[len(cumtmrca[cumtmrca<0.5])]))
print ('Combined 68.3% confidence interval: {} -- {} years'.format(times[len(cumtmrca[cumtmrca<0.1585])],times[len(cumtmrca[cumtmrca<0.8415])]))
print ('Combined 95.0% confidence interval: {} -- {} years'.format(times[len(cumtmrca[cumtmrca<0.025])],times[len(cumtmrca[cumtmrca<0.975])]))
print ('Combined 99.5% confidence interval: {} -- {} years'.format(times[len(cumtmrca[cumtmrca<0.0025])],times[len(cumtmrca[cumtmrca<0.9975])]))

coverage=int(np.random.normal(loc=14.e6,scale=1.e6,size=1))
print ("Coverage=",coverage)

print ("Sample E to D")
tmrca=150
gens=generations(tmrca)
print ("Generations=",gens)
gd_ed,nmut=strs(gens)
print ("GD=",np.sum(np.abs(gd_ed)),"N_mut=",nmut)
snp_ed=snps(tmrca,coverage)
print ("SNPs=",snp_ed)
print ("")

print ("Sample E to C")
tmrca=300
gens=generations(tmrca)
print ("Generations=",gens)
gd_ec,nmut=strs(gens)
print ("GD=",np.sum(np.abs(gd_ec)),"N_mut=",nmut)
snp_ec=snps(tmrca,coverage)
print ("SNPs=",snp_ec)
print ("")

# Compute TMRCA E
ht1=ht0+gd_ed
ht2=ht0+gd_ec
hta=np.ndarray.flatten(scipy.stats.mode(np.array((ht0,ht1,ht2))).mode)
print ("Ex1E",hta)
print ("Ex1D",ht1)
print ("Ex1C",ht2)
output_tmrca,tmrca1,tmrca2=test2(hta,ht1,ht2,snp_ed,snp_ec,coverage,testpaper1,testpaperunc1,testpaperalpha1,testpapertype1,testpaper2,testpaperunc2,testpaperalpha2,testpapertype2,testpaperuncs,testpapertypes)
# Add time the TMRCA of D
tmrca1D=np.convolve(tmrca1,output_tmrca_d,mode='full')[0:len(tmrca1)]
output_tmrca_E=tmrca1D*tmrca2
output_tmrca_E/=np.sum(output_tmrca_E)
cumtmrca=np.cumsum(output_tmrca)
print ('Combined Central estimate:          {} years'.format(times[len(cumtmrca[cumtmrca<0.5])]))
print ('Combined 68.3% confidence interval: {} -- {} years'.format(times[len(cumtmrca[cumtmrca<0.1585])],times[len(cumtmrca[cumtmrca<0.8415])]))
print ('Combined 95.0% confidence interval: {} -- {} years'.format(times[len(cumtmrca[cumtmrca<0.025])],times[len(cumtmrca[cumtmrca<0.975])]))
print ('Combined 99.5% confidence interval: {} -- {} years'.format(times[len(cumtmrca[cumtmrca<0.0025])],times[len(cumtmrca[cumtmrca<0.9975])]))

print ("If paper trail included")
# Add time the TMRCA of D
paper_tmrca=np.zeros(len(tmrca1))
paper_tmrca[150]=1.
tmrca1D=np.convolve(tmrca1,paper_tmrca,mode='full')[0:len(tmrca1)]
output_tmrca=tmrca1D*tmrca2
output_tmrca/=np.sum(output_tmrca)
cumtmrca=np.cumsum(output_tmrca)
print ('Combined Central estimate:          {} years'.format(times[len(cumtmrca[cumtmrca<0.5])]))
print ('Combined 68.3% confidence interval: {} -- {} years'.format(times[len(cumtmrca[cumtmrca<0.1585])],times[len(cumtmrca[cumtmrca<0.8415])]))
print ('Combined 95.0% confidence interval: {} -- {} years'.format(times[len(cumtmrca[cumtmrca<0.025])],times[len(cumtmrca[cumtmrca<0.975])]))
print ('Combined 99.5% confidence interval: {} -- {} years'.format(times[len(cumtmrca[cumtmrca<0.0025])],times[len(cumtmrca[cumtmrca<0.9975])]))

print ("If E is used to constrain D")
tmrca1D=np.convolve(output_tmrca_E,tmrca1,mode='full')[-len(tmrca1):]
output_tmrca_conv=(tmrca1D*output_tmrca_d)
output_tmrca_add=(tmrca1D*output_tmrca_d)
output_tmrca=(output_tmrca_conv/np.sum(output_tmrca_conv)+output_tmrca_add/np.sum(output_tmrca_add))/2.  # See text about why this is a fudge
#output_tmrca/=np.sum(output_tmrca)
cumtmrca=np.cumsum(output_tmrca)
print ('Combined Central estimate:          {} years'.format(times[len(cumtmrca[cumtmrca<0.5])]))
print ('Combined 68.3% confidence interval: {} -- {} years'.format(times[len(cumtmrca[cumtmrca<0.1585])],times[len(cumtmrca[cumtmrca<0.8415])]))
print ('Combined 95.0% confidence interval: {} -- {} years'.format(times[len(cumtmrca[cumtmrca<0.025])],times[len(cumtmrca[cumtmrca<0.975])]))
print ('Combined 99.5% confidence interval: {} -- {} years'.format(times[len(cumtmrca[cumtmrca<0.0025])],times[len(cumtmrca[cumtmrca<0.9975])]))



# Example 2
print ("")
print ("")
print ("--- EXAMPLE 2 ---")
coverage=int(np.random.normal(loc=14.e6,scale=1.e6,size=1))
print ("Coverage=",coverage)

print ("Sample C MacEm to A MacEm")
tmrca=400
gens=generations(tmrca)
print ("Generations=",gens)
gd_ca,nmut=strs(gens)
print ("GD=",np.sum(np.abs(gd_ca)),"N_mut=",nmut)
snp_ca=snps(tmrca,coverage)
print ("SNPs=",snp_ca)
print ("")

print ("Sample C MacEm to B MacEm")
tmrca=400
gens=generations(tmrca)
print ("Generations=",gens)
gd_cb,nmut=strs(gens)
print ("GD=",np.sum(np.abs(gd_cb)),"N_mut=",nmut)
snp_cb=snps(tmrca,coverage)
print ("SNPs=",snp_cb)
print ("")

# Compute TMRCA M
ht1=ht0+gd_ca
ht2=ht0+gd_cb
hta=np.ndarray.flatten(scipy.stats.mode(np.array((ht0,ht1,ht2))).mode)
print ("Ex2M",hta)
print ("Ex2A",ht1)
print ("Ex2B",ht2)
output_tmrca_m,tmrca1,tmrca2=test2(hta,ht1,ht2,snp_ca,snp_cb,coverage,testpaper1,testpaperunc1,testpaperalpha1,testpapertype1,testpaper2,testpaperunc2,testpaperalpha2,testpapertype2,testpaperuncs,testpapertypes)
cumtmrca=np.cumsum(output_tmrca_m)
print ('Combined Central estimate:          {} years'.format(times[len(cumtmrca[cumtmrca<0.5])]))
print ('Combined 68.3% confidence interval: {} -- {} years'.format(times[len(cumtmrca[cumtmrca<0.1585])],times[len(cumtmrca[cumtmrca<0.8415])]))
print ('Combined 95.0% confidence interval: {} -- {} years'.format(times[len(cumtmrca[cumtmrca<0.025])],times[len(cumtmrca[cumtmrca<0.975])]))
print ('Combined 99.5% confidence interval: {} -- {} years'.format(times[len(cumtmrca[cumtmrca<0.0025])],times[len(cumtmrca[cumtmrca<0.9975])]))

coverage=int(np.random.normal(loc=14.e6,scale=1.e6,size=1))
print ("Coverage=",coverage)

print ("Sample F O'En to D O'En")
tmrca=300
gens=generations(tmrca)
print ("Generations=",gens)
gd_fd,nmut=strs(gens)
print ("GD=",np.sum(np.abs(gd_fd)),"N_mut=",nmut)
snp_fd=snps(tmrca,coverage)
print ("SNPs=",snp_fd)
print ("")

print ("Sample F O'En to E O'En")
tmrca=300
gens=generations(tmrca)
print ("Generations=",gens)
gd_fe,nmut=strs(gens)
print ("GD=",np.sum(np.abs(gd_fe)),"N_mut=",nmut)
coverage=int(np.random.normal(loc=14.e6,scale=1.e6,size=1))
print ("Coverage=",coverage)
snp_fe=snps(tmrca,coverage)
print ("SNPs=",snp_fe)
print ("")

# Compute TMRCA N
ht1=ht0+gd_fd
ht2=ht0+gd_fe
hta=np.ndarray.flatten(scipy.stats.mode(np.array((ht0,ht1,ht2))).mode)
print ("Ex2N",hta)
print ("Ex2D",ht1)
print ("Ex2E",ht2)
output_tmrca_n,tmrca1,tmrca2=test2(hta,ht1,ht2,snp_fd,snp_fe,coverage,testpaper1,testpaperunc1,testpaperalpha1,testpapertype1,testpaper2,testpaperunc2,testpaperalpha2,testpapertype2,testpaperuncs,testpapertypes)
cumtmrca=np.cumsum(output_tmrca_n)
print ('Combined Central estimate:          {} years'.format(times[len(cumtmrca[cumtmrca<0.5])]))
print ('Combined 68.3% confidence interval: {} -- {} years'.format(times[len(cumtmrca[cumtmrca<0.1585])],times[len(cumtmrca[cumtmrca<0.8415])]))
print ('Combined 95.0% confidence interval: {} -- {} years'.format(times[len(cumtmrca[cumtmrca<0.025])],times[len(cumtmrca[cumtmrca<0.975])]))
print ('Combined 99.5% confidence interval: {} -- {} years'.format(times[len(cumtmrca[cumtmrca<0.0025])],times[len(cumtmrca[cumtmrca<0.9975])]))

coverage=int(np.random.normal(loc=14.e6,scale=1.e6,size=1))
print ("Coverage=",coverage)

print ("Sample Emn to C MacEm")
tmrca=700
gens=generations(tmrca)
print ("Generations=",gens)
gd_xc,nmut=strs(gens)
print ("GD=",np.sum(np.abs(gd_xc)),"N_mut=",nmut)
snp_xc=snps(tmrca,coverage)
print ("SNPs=",snp_xc)
print ("")

print ("Sample Emn to F O'En")
tmrca=800
gens=generations(tmrca)
print ("Generations=",gens)
gd_xf,nmut=strs(gens)
print ("GD=",np.sum(np.abs(gd_xf)),"N_mut=",nmut)
snp_xf=snps(tmrca,coverage)
print ("SNPs=",snp_xf)
print ("")

# Compute TMRCA Emn
ht1=ht0+gd_xc
ht2=ht0+gd_xf
hta=np.ndarray.flatten(scipy.stats.mode(np.array((ht0,ht1,ht2))).mode)
print ("Ex2MN",hta)
print ("Ex2M",ht1)
print ("Ex2N",ht2)
output_tmrca,tmrca1,tmrca2=test2(hta,ht1,ht2,snp_xc,snp_xf,coverage,testpaper1,testpaperunc1,testpaperalpha1,testpapertype1,testpaper2,testpaperunc2,testpaperalpha2,testpapertype2,testpaperuncs,testpapertypes)
# Add time the TMRCA of C and F
tmrca1=np.convolve(tmrca1,output_tmrca_m,mode='full')[0:len(tmrca1)]
tmrca2=np.convolve(tmrca2,output_tmrca_n,mode='full')[0:len(tmrca2)]
output_tmrca=tmrca1*tmrca2
output_tmrca/=np.sum(output_tmrca)
cumtmrca=np.cumsum(output_tmrca)
print ('Combined Central estimate:          {} years'.format(times[len(cumtmrca[cumtmrca<0.5])]))
print ('Combined 68.3% confidence interval: {} -- {} years'.format(times[len(cumtmrca[cumtmrca<0.1585])],times[len(cumtmrca[cumtmrca<0.8415])]))
print ('Combined 95.0% confidence interval: {} -- {} years'.format(times[len(cumtmrca[cumtmrca<0.025])],times[len(cumtmrca[cumtmrca<0.975])]))
print ('Combined 99.5% confidence interval: {} -- {} years'.format(times[len(cumtmrca[cumtmrca<0.0025])],times[len(cumtmrca[cumtmrca<0.9975])]))

print ("If paper trail included")
# Add time the TMRCA of C and F
paper_tmrca1=np.zeros(len(tmrca1))
paper_tmrca1[400]=1.
tmrca1M=np.convolve(tmrca1,paper_tmrca,mode='full')[0:len(tmrca1)]
paper_tmrca2=np.zeros(len(tmrca2))
paper_tmrca2[300]=1.
tmrca2N=np.convolve(tmrca2,paper_tmrca,mode='full')[0:len(tmrca2)]
output_tmrca=tmrca1M*tmrca2N
output_tmrca/=np.sum(output_tmrca)
cumtmrca=np.cumsum(output_tmrca)
print ('Combined Central estimate:          {} years'.format(times[len(cumtmrca[cumtmrca<0.5])]))
print ('Combined 68.3% confidence interval: {} -- {} years'.format(times[len(cumtmrca[cumtmrca<0.1585])],times[len(cumtmrca[cumtmrca<0.8415])]))
print ('Combined 95.0% confidence interval: {} -- {} years'.format(times[len(cumtmrca[cumtmrca<0.025])],times[len(cumtmrca[cumtmrca<0.975])]))
print ('Combined 99.5% confidence interval: {} -- {} years'.format(times[len(cumtmrca[cumtmrca<0.0025])],times[len(cumtmrca[cumtmrca<0.9975])]))



# Example 3
print ("")
print ("")
print ("--- EXAMPLE 3 ---")
coverage=int(np.random.normal(loc=14.e6,scale=1.e6,size=1))
print ("Coverage=",coverage)

print ("Sample D. Smith to A. Smith")
tmrca=600
gens=generations(tmrca)
print ("Generations=",gens)
gd_da,nmut=strs(gens)
print ("GD=",np.sum(np.abs(gd_da)),"N_mut=",nmut)
snp_da=snps(tmrca,coverage)
print ("SNPs=",snp_da)
print ("")

print ("Sample D. Smith to B. Smith")
tmrca=600
gens=generations(tmrca)
print ("Generations=",gens)
gd_db,nmut=strs(gens)
print ("GD=",np.sum(np.abs(gd_db)),"N_mut=",nmut)
snp_db=snps(tmrca,coverage)
print ("SNPs=",snp_db)
print ("")

print ("Sample D. Smith to C. Smith")
tmrca=600
gens=generations(tmrca)
print ("Generations=",gens)
gd_dc,nmut=strs(gens)
print ("GD=",np.sum(np.abs(gd_dc)),"N_mut=",nmut)
snp_dc=snps(tmrca,coverage)
print ("SNPs=",snp_dc)
print ("")

# Compute TMRCA UK
ht1=ht0+gd_da
ht2=ht0+gd_db
ht3=ht0+gd_dc
hta=np.ndarray.flatten(scipy.stats.mode(np.array((ht0,ht1,ht2,ht3))).mode)
print ("Ex3UK",hta)
print ("Ex3A",ht1)
print ("Ex3B",ht2)
print ("Ex3C",ht3)
output_tmrca,tmrca1,tmrca2=test2(hta,ht1,ht2,snp_da,snp_db,coverage,testpaper1,testpaperunc1,testpaperalpha1,testpapertype1,testpaper2,testpaperunc2,testpaperalpha2,testpapertype2,testpaperuncs,testpapertypes)
output_tmrca,tmrca1,tmrca3=test2(hta,ht1,ht3,snp_da,snp_dc,coverage,testpaper1,testpaperunc1,testpaperalpha1,testpapertype1,testpaper2,testpaperunc2,testpaperalpha2,testpapertype2,testpaperuncs,testpapertypes)
output_tmrca_uk=tmrca1*tmrca2*tmrca3
output_tmrca_uk/=np.sum(output_tmrca_uk)
cumtmrca=np.cumsum(output_tmrca_uk)
print ('Combined Central estimate:          {} years'.format(times[len(cumtmrca[cumtmrca<0.5])]))
print ('Combined 68.3% confidence interval: {} -- {} years'.format(times[len(cumtmrca[cumtmrca<0.1585])],times[len(cumtmrca[cumtmrca<0.8415])]))
print ('Combined 95.0% confidence interval: {} -- {} years'.format(times[len(cumtmrca[cumtmrca<0.025])],times[len(cumtmrca[cumtmrca<0.975])]))
print ('Combined 99.5% confidence interval: {} -- {} years'.format(times[len(cumtmrca[cumtmrca<0.0025])],times[len(cumtmrca[cumtmrca<0.9975])]))

coverage=int(np.random.normal(loc=14.e6,scale=1.e6,size=1))
print ("Coverage=",coverage)

print ("Sample G. Schmidt to E. Schmidt")
tmrca=600
gens=generations(tmrca)
print ("Generations=",gens)
gd_ge,nmut=strs(gens)
print ("GD=",np.sum(np.abs(gd_ge)),"N_mut=",nmut)
snp_ge=snps(tmrca,coverage)
print ("SNPs=",snp_ge)
print ("")

print ("Sample G. Schmidt to F. Schmidt")
tmrca=600
gens=generations(tmrca)
print ("Generations=",gens)
gd_gf,nmut=strs(gens)
print ("GD=",np.sum(np.abs(gd_gf)),"N_mut=",nmut)
snp_gf=snps(tmrca,coverage)
print ("SNPs=",snp_gf)
print ("")

# Compute TMRCA DE
ht1=ht0+gd_ge
ht2=ht0+gd_gf
hta=np.ndarray.flatten(scipy.stats.mode(np.array((ht0,ht1,ht2))).mode)
print ("Ex3DE",hta)
print ("Ex3E",ht1)
print ("Ex3F",ht2)
output_tmrca_de,tmrca1,tmrca2=test2(hta,ht1,ht2,snp_ge,snp_gf,coverage,testpaper1,testpaperunc1,testpaperalpha1,testpapertype1,testpaper2,testpaperunc2,testpaperalpha2,testpapertype2,testpaperuncs,testpapertypes)
cumtmrca=np.cumsum(output_tmrca_de)
print ('Combined Central estimate:          {} years'.format(times[len(cumtmrca[cumtmrca<0.5])]))
print ('Combined 68.3% confidence interval: {} -- {} years'.format(times[len(cumtmrca[cumtmrca<0.1585])],times[len(cumtmrca[cumtmrca<0.8415])]))
print ('Combined 95.0% confidence interval: {} -- {} years'.format(times[len(cumtmrca[cumtmrca<0.025])],times[len(cumtmrca[cumtmrca<0.975])]))
print ('Combined 99.5% confidence interval: {} -- {} years'.format(times[len(cumtmrca[cumtmrca<0.0025])],times[len(cumtmrca[cumtmrca<0.9975])]))

coverage=int(np.random.normal(loc=14.e6,scale=1.e6,size=1))
print ("Coverage=",coverage)

print ("Sample MRCA to Smith")
tmrca=3400
gens=generations(tmrca)
print ("Generations=",gens)
gd_xuk,nmut=strs(gens)
print ("GD=",np.sum(np.abs(gd_xuk)),"N_mut=",nmut)
snp_xuk=snps(tmrca,coverage)
print ("SNPs=",snp_xuk)
print ("")

print ("Sample MRCA to Schmidt")
tmrca=3400
gens=generations(tmrca)
print ("Generations=",gens)
gd_xde,nmut=strs(gens)
print ("GD=",np.sum(np.abs(gd_xde)),"N_mut=",nmut)
snp_xde=snps(tmrca,coverage)
print ("SNPs=",snp_xde)
print ("")

# Compute TMRCA
ht1=ht0+gd_xuk
ht2=ht0+gd_xde
hta=np.ndarray.flatten(scipy.stats.mode(np.array((ht0,ht1,ht2))).mode)
print ("Ex3CA",hta)
print ("Ex3UK",ht1)
print ("Ex3DE",ht2)
output_tmrca,tmrca1,tmrca2=test2(hta,ht1,ht2,snp_xuk,snp_xde,coverage,testpaper1,testpaperunc1,testpaperalpha1,testpapertype1,testpaper2,testpaperunc2,testpaperalpha2,testpapertype2,testpaperuncs,testpapertypes)
# Add time the TMRCA of C and F
tmrca1=np.convolve(tmrca1,output_tmrca_uk,mode='full')[0:len(tmrca1)]
tmrca2=np.convolve(tmrca2,output_tmrca_de,mode='full')[0:len(tmrca2)]
output_tmrca=tmrca1*tmrca2
output_tmrca/=np.sum(output_tmrca)
cumtmrca=np.cumsum(output_tmrca)
print ('Combined Central estimate:          {} years'.format(times[len(cumtmrca[cumtmrca<0.5])]))
print ('Combined 68.3% confidence interval: {} -- {} years'.format(times[len(cumtmrca[cumtmrca<0.1585])],times[len(cumtmrca[cumtmrca<0.8415])]))
print ('Combined 95.0% confidence interval: {} -- {} years'.format(times[len(cumtmrca[cumtmrca<0.025])],times[len(cumtmrca[cumtmrca<0.975])]))
print ('Combined 99.5% confidence interval: {} -- {} years'.format(times[len(cumtmrca[cumtmrca<0.0025])],times[len(cumtmrca[cumtmrca<0.9975])]))





def test2x(ht0,ht1,ht2,testsnpdist1,testsnpdist2,testcombcov,testpaper1,testpaperunc1,testpaperalpha1,testpapertype1,testpaper2,testpaperunc2,testpaperalpha2,testpapertype2,tystpaperuncs,testpapertypes):
	# Test data

	# Choose what tests to include
	do_strs=1
	do_snps=0
	do_indels=0
	do_paper=0
		
	if (do_paper>0):
		tmrca_paper=np.ones(len(dates))
		# TMRCA from paper results from the conjunction of TMRCA 1 & 2,
		# so it is the maximum probability of being before this date.
		# This allows for one line to join the other before the oldest
		# most-distant known ancestor.
		tmrca_paper*=np.maximum(paperpdf(dates,testpapertype1,testpaper1,testpaperunc1,testpaperalpha1),paperpdf(dates,testpapertype2,testpaper2,testpaperunc1,testpaperalpha1))
		# Account for shared surname
		tmrca_paper*=paperpdf(dates,testpapertypes,testpapers,testpaperuncs)
		#np.savetxt("tpaper.dat", np.c_[dates,tmrca_paper], delimiter=' ')
	else:
		tmrca_paper=np.ones(len(dates))
	
	if (do_strs>0):
		# Calculate TMRCA from STRs
		#trace(0, 'Calculating STR-based TMRCA')
		tstr1=strage(times,pdf_str_t,ht0,ht1)
		#np.savetxt("tstr1.dat", np.c_[times,tstr1], delimiter=' ')
		tstr2=strage(times,pdf_str_t,ht0,ht2)
		#np.savetxt("tstr2.dat", np.c_[times,tstr2], delimiter=' ')
		# Combine these TMRCAs
		tmrca_str=tstr1*tstr2
		# Normalise PDF
		psum=sum(tmrca_str)
		tmrca_str/=psum
		np.savetxt("tstr.dat", np.c_[times,tmrca_str], delimiter=' ')
		# Get confidence intervals
		cumtmrca=np.cumsum(tmrca_str)
		#print ('STR Central estimate:          {} years'.format(times[len(cumtmrca[cumtmrca<0.5])]))
		#print ('STR 68.3% confidence interval: {} -- {} years'.format(times[len(cumtmrca[cumtmrca<0.1585])],times[len(cumtmrca[cumtmrca<0.8415])]))
		#print ('STR 95.0% confidence interval: {} -- {} years'.format(times[len(cumtmrca[cumtmrca<0.025])],times[len(cumtmrca[cumtmrca<0.975])]))
		#print ('STR 99.5% confidence interval: {} -- {} years'.format(times[len(cumtmrca[cumtmrca<0.0025])],times[len(cumtmrca[cumtmrca<0.9975])]))
	else:
		tmrca_str=np.ones(len(times))
		tstr1=np.ones(len(times))
		tstr2=np.ones(len(times))

	if (do_snps>0):
		#trace(0, 'Calculating SNP-based TMRCA')
		# Calculate TMRCA from SNPs
		stimes=np.arange(0,maxsnps,snpstep)
		tsnpm1=scipy.stats.poisson.pmf(testsnpdist1,stimes)
		tsnpm2=scipy.stats.poisson.pmf(testsnpdist2,stimes)
		tsnpm=tsnpm1*tsnpm2
		# Convert to physical timescale
		tmrca_snp=snptmconv(times,stimes,tsnpm,testcombcov,musnp,musnpunc)
		tsnp1=snptmconv(times,stimes,tsnpm1,testcombcov,musnp,musnpunc)
		#np.savetxt("tsnp1.dat", np.c_[times,tsnp1], delimiter=' ')
		tsnp2=snptmconv(times,stimes,tsnpm2,testcombcov,musnp,musnpunc)
		#np.savetxt("tsnp2.dat", np.c_[times,tsnp2], delimiter=' ')
		# Normalise PDF
		psum=sum(tmrca_snp)
		tmrca_snp/=psum
		np.savetxt("tsnp.dat", np.c_[times,tmrca_snp], delimiter=' ')
		# Get confidence intervals
		cumtmrca=np.cumsum(tmrca_snp)
		#print ('SNP Central estimate:          {} years'.format(times[len(cumtmrca[cumtmrca<0.5])]))
		#print ('SNP 68.3% confidence interval: {} -- {} years'.format(times[len(cumtmrca[cumtmrca<0.1585])],times[len(cumtmrca[cumtmrca<0.8415])]))
		#print ('SNP 95.0% confidence interval: {} -- {} years'.format(times[len(cumtmrca[cumtmrca<0.025])],times[len(cumtmrca[cumtmrca<0.975])]))
		#print ('SNP 99.5% confidence interval: {} -- {} years'.format(times[len(cumtmrca[cumtmrca<0.0025])],times[len(cumtmrca[cumtmrca<0.9975])]))
	else:
		tmrca_snp=np.ones(len(times))
		tsnp1=np.ones(len(times))
		tsnp2=np.ones(len(times))
	
	if (do_indels>0):
		trace(0, 'Calculating indel-based TMRCA')
		# Calculate TMRCA from indels
		stimes=np.arange(0,maxsnps,snpstep)
		tsnpm1=scipy.stats.poisson.pmf(testindeldist1,stimes)
		tsnpm2=scipy.stats.poisson.pmf(testindeldist2,stimes)
		tsnpm=tsnpm1*tsnpm2
		# Convert to physical timescale
		tmrca_indel=snptmconv(times,stimes,tsnpm,testcombcov,muindel,muindelunc)
		tindel1=snptmconv(times,stimes,tsnpm,testcombcov,musnp,musnpunc)
		#np.savetxt("tindel1.dat", np.c_[times,tindel1], delimiter=' ')
		tindel2=snptmconv(times,stimes,tsnpm,testcombcov,musnp,musnpunc)
		#np.savetxt("tindel2.dat", np.c_[times,tindel2], delimiter=' ')
		# Normalise PDF
		psum=sum(tmrca_indel)
		tmrca_indel/=psum
		np.savetxt("tindel.dat", np.c_[times,tmrca_indel], delimiter=' ')
		# Get confidence intervals
		cumtmrca=np.cumsum(tmrca_indel)
		#print ('Indel Central estimate:          {} years'.format(times[len(cumtmrca[cumtmrca<0.5])]))
		#print ('Indel 68.3% confidence interval: {} -- {} years'.format(times[len(cumtmrca[cumtmrca<0.1585])],times[len(cumtmrca[cumtmrca<0.8415])]))
		#print ('Indel 95.0% confidence interval: {} -- {} years'.format(times[len(cumtmrca[cumtmrca<0.025])],times[len(cumtmrca[cumtmrca<0.975])]))
		#print ('Indel 99.5% confidence interval: {} -- {} years'.format(times[len(cumtmrca[cumtmrca<0.0025])],times[len(cumtmrca[cumtmrca<0.9975])]))
	else:
		tmrca_indel=np.ones(len(times))
		tindel1=np.ones(len(times))
		tindel2=np.ones(len(times))

	tmrca=tmrca_str*tmrca_snp*tmrca_indel
	tmrca1=tstr1*tsnp1*tindel1
	tmrca2=tstr2*tsnp2*tindel2
	# Normalise PDF
	tmrca/=np.sum(tmrca)
	tmrca1/=np.sum(tmrca1)
	tmrca2/=np.sum(tmrca2)
	#np.savetxt("tmrca.dat", np.c_[times,tmrca], delimiter=' ')
	# Get confidence intervals
	cumtmrca=np.cumsum(tmrca)
	#print (cumtmrca)
	#print ('Combined Central estimate:          {} years'.format(times[len(cumtmrca[cumtmrca<0.5])]))
	#print ('Combined 68.3% confidence interval: {} -- {} years'.format(times[len(cumtmrca[cumtmrca<0.1585])],times[len(cumtmrca[cumtmrca<0.8415])]))
	#print ('Combined 95.0% confidence interval: {} -- {} years'.format(times[len(cumtmrca[cumtmrca<0.025])],times[len(cumtmrca[cumtmrca<0.975])]))
	#print ('Combined 99.5% confidence interval: {} -- {} years'.format(times[len(cumtmrca[cumtmrca<0.0025])],times[len(cumtmrca[cumtmrca<0.9975])]))
	# Convert to dates
	dt=dt0/sqrt(2) # Average birth date uncertainty is reduced by factor sqrt(2) for two testers
	st=sqrt(st0**2+dt**2) # Add uncertainty in the mean
	dmrca=ages2dates(tmrca,dates,times,timestep,nowtime,t0,st) # Convert TMRCA to dates
	#print (len(dmrca),len(tmrca_paper))
	dmrca*=tmrca_paper # Add in paper-trail genealogy limits
	# Normalise PDF
	psum=sum(dmrca)
	dmrca/=psum
	#np.savetxt("dmrca.dat", np.c_[dates,dmrca], delimiter=' ')
	# Get confidence intervals
	cumdmrca=np.cumsum(dmrca)
	#print (cumdmrca)
	#centiles=[0.5,0.1585,0.8415,0.025,0.975,0.0025,0.9975]
	#estimators=dates[len(cumdmrca[cumdmrca<centiles])]
	#ad=(dates>0)
	#print ('Combined Central estimate:          {} CE'.format(dates[len(cumdmrca[cumdmrca<0.5])]))
	#print ('Combined 68.3% confidence interval: {} -- {} CE'.format(dates[len(cumdmrca[cumdmrca<0.8415])],dates[len(cumdmrca[cumdmrca<0.1585])]))
	#print ('Combined 95.0% confidence interval: {} -- {} CE'.format(dates[len(cumdmrca[cumdmrca<0.975])],dates[len(cumdmrca[cumdmrca<0.025])]))
	#print ('Combined 99.5% confidence interval: {} -- {} CE'.format(dates[len(cumdmrca[cumdmrca<0.9975])],dates[len(cumdmrca[cumdmrca<0.0025])]))

	return tmrca,tmrca1,tmrca2

output_tmrca,tmrca1,tmrca2=test2x(hta,ht1,ht2,snp_xuk,snp_xde,coverage,testpaper1,testpaperunc1,testpaperalpha1,testpapertype1,testpaper2,testpaperunc2,testpaperalpha2,testpapertype2,testpaperuncs,testpapertypes)
# Add time the TMRCA of C and F
tmrca1=np.convolve(tmrca1,output_tmrca_uk,mode='full')[0:len(tmrca1)]
tmrca2=np.convolve(tmrca2,output_tmrca_de,mode='full')[0:len(tmrca2)]
output_tmrca=tmrca1*tmrca2
output_tmrca/=np.sum(output_tmrca)
cumtmrca=np.cumsum(output_tmrca)
print ('Combined Central estimate:          {} years'.format(times[len(cumtmrca[cumtmrca<0.5])]))
print ('Combined 68.3% confidence interval: {} -- {} years'.format(times[len(cumtmrca[cumtmrca<0.1585])],times[len(cumtmrca[cumtmrca<0.8415])]))
print ('Combined 95.0% confidence interval: {} -- {} years'.format(times[len(cumtmrca[cumtmrca<0.025])],times[len(cumtmrca[cumtmrca<0.975])]))
print ('Combined 99.5% confidence interval: {} -- {} years'.format(times[len(cumtmrca[cumtmrca<0.0025])],times[len(cumtmrca[cumtmrca<0.9975])]))
