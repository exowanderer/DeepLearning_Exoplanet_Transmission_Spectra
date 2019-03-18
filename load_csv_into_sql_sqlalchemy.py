# from https://stackoverflow.com/questions/31394998/using-sqlalchemy-to-load-csv-file-into-a-database

from numpy import genfromtxt, nan
from time import time
from datetime import datetime
from sqlalchemy import Column, Integer, Float, Date, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

Base = declarative_base()

class Comet(Base):
    # Tell SQLAlchemy what the table name is and if there's any table-specific arguments it should know about
    __tablename__ = 'Comets'
    __table_args__ = {'sqlite_autoincrement': True}
    
    # tell SQLAlchemy the name of column and its attributes:
    id = Column(Integer, primary_key=True, nullable=False)
    comet_name = Column(String)
    xcenter = Column(Float)
    ycenter = Column(Float)
    fwhm = Column(Float)
    
    # Science Frame Entries
    sci_img = Column(String)
    sci_ctlg = Column(String)
    sci_date_string = Column(String)
    sci_date = Column(String)
    sci_field = Column(Integer)
    sci_unique_id = Column(Integer) # , unique=True
    sci_filter = Column(Integer)
    sci_p_num = Column(Integer)
    sci_channel = Column(Integer)
    
    # Reference Frame Entries
    ref_img = Column(String)
    ref_ctlg = Column(String)
    ref_date_string = Column(String)
    ref_date = Column(String)
    ref_field = Column(Integer)
    ref_unique_id = Column(Integer) # , unique=True
    ref_filter = Column(Integer)
    ref_p_num = Column(Integer)
    ref_channel = Column(Integer)

if __name__ == "__main__":
    # from pandas import read_csv
    from glob import glob
    from tqdm import tqdm
    from time import time
    import numpy as np

    ''' Input Data '''
    wavelengths = None
    not_jwst = None
    verbose = True

    if verbose: print("[INFO] Load data from harddrive.")

    spectra_filenames = glob('transmission_spectral_files/trans*')
    spectral_grid = {}

    for fname in tqdm(spectral_filenames):
        key = '_'.join(fname.split('/')[-1].split('_')[1:7])
        info_now = np.loadtxt(fname)
        if wavelengths is None: wavelengths = info_now[:,0]
        if not_jwst is None: not_jwst = wavelengths < 5.0
        spectral_grid[key] = info_now[:,1][not_jwst]

    if verbose: print("[INFO] Assigning input values onto `labels` and `features`")

    n_waves = not_jwst.sum()
    labels = np.zeros((len(spectral_filenames), n_waves))
    features = np.zeros((len(spectral_filenames), len(key.split('_'))))

    for k, (key,val) in enumerate(spectral_grid.items()): 
        labels[k] = val
        features[k] = np.array(key.split('_')).astype(float)

    if verbose: print("[INFO] Computing train test split over indices "
                        "with shuffling")

    start = time()
    
    #Create the database
    engine = create_engine('sqlite:///exoplanet_input_spectral_catalog.db')
    Base.metadata.create_all(engine)
    
    #Create the session
    session = sessionmaker()
    session.configure(bind=engine)
    s = session()
    
    try:
        # for datum in data.values:
        for k in range(len(data)):
            
            datum = data.iloc[k]
            
            if isinstance(datum['sci_img'], str) and isinstance(datum['ref_img'], str):
                _, sci_date, _, _, _, sci_field, sci_unique_id, sci_filter, sci_p_num, sci_channel = datum['sci_img'].split('_')
                _, ref_field, ref_filter, ref_channel, ref_unique_id, ref_p_num, _ = datum['ref_img'].split('_')
                
                sci_channel = sci_channel.replace('.fits', '')
            else:
                sci_date, sci_field, sci_unique_id, sci_filter, sci_p_num, sci_channel = [nan]*6
                ref_field, ref_filter, ref_channel, ref_unique_id, ref_p_num = [nan]*5
            
            record = Comet(**{
                'comet_name' : datum['comet name'], # datetime.strptime(datum[0], '%d-%b-%y').date(),
                'xcenter' : datum['xc'],
                'ycenter' : datum['yc'],
                'fwhm' : datum['fwhm'],
                
                'sci_img' : datum['sci_img'],
                'sci_ctlg' : datum['sci_ctlg'],
                'sci_date' : sci_date,
                'sci_field' : sci_field,
                'sci_unique_id' : sci_unique_id,
                'sci_filter' : sci_filter,
                'sci_p_num' : sci_p_num,
                'sci_channel' : sci_channel,
                
                'ref_img' : datum['ref_img'],
                'ref_ctlg' : datum['ref_ctlg'],
                'ref_field' : ref_field,
                'ref_unique_id' : ref_unique_id,
                'ref_filter' : ref_filter,
                'ref_p_num' : ref_p_num,
                'ref_channel' : ref_channel
            })
            
            s.add(record) #Add all the records
        
        s.commit() # Attempt to commit all the records
    except:
        s.rollback() # Rollback the changes on error
    finally:
        s.close() # Close the connection
    print("Time elapsed: {} seconds".format(time() - start)) #0.091s