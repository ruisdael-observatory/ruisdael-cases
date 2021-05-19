import numpy as np

class BOFEK_info:
    def __init__(self):
        #
        # Lookup table name -> index in `van_genuchten_parameters.{tbl/nc}`
        #
        lookup = {}
        for i in range(1,19):
            lookup['B{}'.format(i)] = i+5
        for i in range(1,19):
            lookup['O{}'.format(i)] = i+23

        #
        # Read the CSV table with BOFEK2012 info
        #
        soil_id_tbl, z_top_tbl, z_bot_tbl = np.genfromtxt(
                'BOFEK2012_profielen_versie2_1.csv', skip_header=1, usecols=[0,7,8],
                delimiter=',', unpack=True)
        soil_id_tbl = soil_id_tbl.astype(int)

        soil_code_tbl = np.genfromtxt(
                'BOFEK2012_profielen_versie2_1.csv', skip_header=1, usecols=29,
                delimiter=',', unpack=True, dtype=str)

        #
        # Generate 2D Numpy arrays from flat BOFEK2012 table
        #
        unique_ids = np.unique(soil_id_tbl)
        n_types = unique_ids.size
        max_layers = 8

        #
        # Main external output
        #
        # Profiles of O1/B2/O5 etc. codes:
        self.OB_code = np.zeros((n_types, max_layers), dtype='U3')

        # Index of O1/B2/O5 etc. codes in `van_genuchten_parameters.nc` table:
        self.lookup_index = np.zeros((n_types, max_layers), dtype=int)-1

        # Top, bottom and center of soil layers:
        self.z_top = np.zeros((n_types, max_layers), dtype=float)-1
        self.z_bot = np.zeros((n_types, max_layers), dtype=float)-1
        self.z_mid = np.zeros((n_types, max_layers), dtype=float)-1

        # Max layers per soil type:
        self.n_layers = np.zeros(n_types, dtype=int)

        # BOFEK ID/code of soil types:
        self.soil_id = np.zeros(n_types, dtype=int)

        # Lookup table BOFEK ID/code -> index in `soil_id`, `z_top`, etc. tables:
        self.soil_id_lu = np.zeros(unique_ids.max()+1, dtype=int)-9999

        ii = 0
        kk = 0
        prev_code = soil_id_tbl[0]
        for i in range(len(soil_id_tbl)):
            if soil_id_tbl[i] != prev_code:
                ii += 1
                kk  = 0

            self.OB_code[ii,kk] = soil_code_tbl[i]
            self.z_top[ii,kk] = z_top_tbl[i]
            self.z_bot[ii,kk] = z_bot_tbl[i]
            self.soil_id[ii] = soil_id_tbl[i]

            prev_code = soil_id_tbl[i]
            kk += 1

        for i in range(n_types):
            self.n_layers[i] = np.argmax(self.z_top[i,:]==-1)

        for i in range(n_types):
            for j in range(self.n_layers[i]):
                self.lookup_index[i,j] = lookup[self.OB_code[i,j].strip()]
                self.z_mid[i,j] = 0.5*(self.z_top[i,j] + self.z_bot[i,j])

        for i in range(n_types):
            self.soil_id_lu[self.soil_id[i]] = i


    def get_info(self, bf_code):
        ii = self.soil_id_lu[bf_code]
        print(self.OB_code[ii])
        print(self.lookup_index[ii])


if __name__ == '__main__':
    """ Just for testing... """

    bf = BOFEK_info()
