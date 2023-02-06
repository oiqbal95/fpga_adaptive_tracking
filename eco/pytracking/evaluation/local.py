from pytracking.evaluation.environment import EnvSettings

def local_env_settings():
    settings = EnvSettings()

    # Set your local paths here.

    settings.davis_dir = ''
    settings.got10k_path = ''
    settings.got_packed_re5sults_path = ''
    settings.got_reports_path = ''
    settings.lasot_path = '/home/ame/Adaptive_Subsampling/LaSOT/Main_Dataset'
    settings.network_path = 'pytracking/parameter/'    # Where tracking networks are stored.
    settings.nfs_path = ''
    settings.otb_path = 'dataset'
    settings.result_plot_path = 'dataset/result_plots/'
    settings.results_path = 'dataset/pytracking/tracking_results/'    # Where to store tracking results
    settings.segmentation_path = 'dataset/segmentation_results/'
    settings.tn_packed_results_path = ''
    settings.tpl_path = ''
    settings.trackingnet_path = '/home/ame/Adaptive_Subsampling/TrackingNet'
    settings.uav_path = ''
    settings.vot_path = ''
    settings.youtubevos_dir = ''

    return settings

