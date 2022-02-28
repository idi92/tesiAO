import os


def data_root_dir():
    import pkg_resources

    dataroot = pkg_resources.resource_filename(
        'tesi_ao',
        'data')
    return dataroot


def file_name_mcl(tag):
    rootDir = data_root_dir()
    return os.path.join(rootDir,
                        'mcl',
                        '%s.fits' % tag)
