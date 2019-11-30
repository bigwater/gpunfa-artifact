import matplotlib
matplotlib.use('agg')

from helpers import *
from ap_estimator_ideal import *

cfg_name_map = {
    'output_infant_infant_0_1000000_1000_256_256' : 'iNFAnt',
    'output_newtran_obat2_0_1000000_1000_256_256' : 'NT',
    'output_newtran_obat_MC_0_1000000_1000_256_256' : 'NT-Mac',
    'output_hotstarttt_hotstart_aa_0_1000000_1000_256_256_1280' : 'HotStartTT',
    'output_hotstart_hotstart_ea_0_1000000_1000_256_256' : 'HotStart-Mac',
    'output_hotstart_hotstart_ea_no_MC2_0_1000000_1000_256_256' : 'HotStart',
    'output_nfacg_ppopp12_0_1000000_1000_256_256' : 'NFA-CG',
    'AP_ideal': 'AP_ideal',
    'AP': 'AP'
}

cfg_rmap = {}

cfg_order = [
'AP',
'AP_ideal',
'iNFAnt',
'NT',
'NT-Mac',
'NFA-CG',
'HotStartTT',
'HotStart',
'HotStart-Mac'
]


app_name_map = {
    'Brill' : 'Brill',
    'ClamAV' : 'CAV',
    'CRISPR_CasOFFinder' : 'CRSPR1',
    'CRISPR_CasOT' : 'CRSPR2',
    'APPRNG_4sided' : 'APPRNG',
    'EntityResolution' : 'ER',
    'Hamming_l18d3' : 'HM',
    'Levenshtein_l19d3' : 'LV',
    'Protomata' : 'Pro',
    'RandomForest_20_400_200' : 'RF',
    'SeqMatch_w6p6' : 'SeqMat',
    'Snort' : 'Snort',
    'YARA' : 'YARA',
    'Bro217' : 'Bro',
    'ExactMath' : 'EM',
    'Ranges05' : 'Rg05',
    'Ranges1' : 'Rg1',
    'TCP' : 'TCP',
    'PowerEN' : 'PEN'
}

app_rmap = {}

apps_original = [
    'Brill',
    'CAV',
    'CRSPR1',
    'CRSPR2',
    
    'ER',
    'HM',
    'LV',
    'Pro',
    
    
    'Snort',
    'YARA',
    'Bro',
    'EM',
    'Rg05',
    'Rg1',
    'TCP',
    'PEN'
]

app_order = []



if __name__ == '__main__':
    app_list = get_app_order()
    #print(app_list)
    for app, tmp in app_list:
        if not app in app_name_map:
            continue

        if app_name_map[app] in apps_original:
            app_order.append(app_name_map[app])

    for k in app_name_map:
        app_rmap[app_name_map[k]] = k

    for k in cfg_name_map:
        cfg_rmap[cfg_name_map[k]] = k

    data1 = readFromFile('avgres_throughput.txt', 'throughput')

    data1['AP_ideal'] = get_throughput(49152, 1000, 1000)
    data1['AP'] = get_throughput(49152, 1000, 1000, False)

    select_apps_in_ds(data1, [app_rmap[a] for a in app_order])

    plt, ax = plot(data1, app_order, app_rmap, cfg_order, cfg_rmap, storebar1name='abs_throughput.csv', geo_mean=False)

    


