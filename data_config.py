
class DataConfig:
    data_name = ""
    root_dir = ""
    label_transform = "norm"
    def get_data_config(self, data_name):
        self.data_name = data_name
        if data_name == 'LEVIR':
            self.label_transform = "norm"
            #self.root_dir = '/media/lidan/ssd2/CDData/LEVIR-CD256/'
            self.root_dir = 'LEVIR'
        elif data_name == 'data':
            self.root_dir = 'data'
        elif data_name == 'LEVIR-test':
            self.root_dir = 'LEVIR-test'
        elif data_name == 'DSIFN':
            self.label_transform = "norm"
            self.root_dir = 'DSIFN'
        elif data_name == 'WHU256':
            self.label_transform = "norm"
            self.root_dir = 'WHU-256'
        elif data_name == 'WHU':
            self.label_transform = "norm"
            self.root_dir = 'WHU'
        elif data_name == 'img5':
            self.root_dir = 'NewMCD/img5'
        elif data_name == 'img6':
            self.root_dir = 'NewMCD/img6'
        elif data_name == 'img7':
            self.root_dir = 'NewMCD/img7'
        elif data_name == 'img10':
            self.root_dir = 'NewMCD/img10'
        elif data_name == 'HCDv3':
            self.root_dir = 'HCDv3'
        elif data_name == 'HCDv4':
            self.root_dir = 'HCDv4'
        else:
            raise TypeError('%s has not defined' % data_name)
        return self


if __name__ == '__main__':
    data = DataConfig().get_data_config(data_name='LEVIR')
    print(data.data_name)
    print(data.root_dir)
    print(data.label_transform)

