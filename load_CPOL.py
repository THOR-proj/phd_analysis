filenames = cf.CPOL_files_from_datetime_list(
            np.arange(np.datetime64('2002-11-01 00:00'), 
                              np.datetime64('2003-04-01 00:00'), 
                                            np.timedelta64(10, 'm'))
            )[0]

filenames_b = cf.CPOL_files_from_datetime_list(
            np.arange(np.datetime64('2003-11-01 00:00'), 
                              np.datetime64('2004-04-01 00:00'), 
                                            np.timedelta64(10, 'm'))
            )[0]

filenames_c = cf.CPOL_files_from_datetime_list(
            np.arange(np.datetime64('2004-11-01 00:00'), 
                              np.datetime64('2005-04-01 00:00'), 
                                            np.timedelta64(10, 'm'))
            )[0]

filenames_d = cf.CPOL_files_from_datetime_list(
            np.arange(np.datetime64('2005-11-01 00:00'), 
                              np.datetime64('2006-04-01 00:00'), 
                                            np.timedelta64(10, 'm'))
            )[0]

filenames_e = cf.CPOL_files_from_datetime_list(
            np.arange(np.datetime64('2006-11-01 00:00'), 
                              np.datetime64('2007-04-01 00:00'), 
                                            np.timedelta64(10, 'm'))
            )[0]

filenames = filenames_a + filenames_b + filenames_c + filenames_d + filenames_e
