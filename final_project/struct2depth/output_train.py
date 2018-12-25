import glob

dataset_dir = "/home/gaofeng/datasets/processed_data/kitti/"
test_files = "/home/gaofeng/workspace/ai_2018fall/final_project/struct2depth/test_files_eigen.txt"

def get_test_files():
    f = open(test_files, 'r')
    lines = f.readlines()
    test_files_list = []
    for line in lines:
        seqname = line.split('/')[1]
        lr = line.split('/')[2].split('_')[-1]
        new_seqname = seqname + '_' + lr
        frame_id = line.split('/')[-1].split('.')[0]
        test_files_list.append(new_seqname + ' ' + frame_id)
    return test_files_list

def main():
    test_files_list = get_test_files()
    f = open(dataset_dir + 'train.txt', 'w')
    for d in glob.glob(dataset_dir + '/*/'):
        seqname = d.split('/')[-2]
        for d2 in glob.glob(d + '/*.png'):
            frameid = d2.split('/')[-1].split('.')[0]
            line = seqname + ' ' + frameid
            if line in test_files_list or 'seg' in frameid:
                continue
            f.write(line + '\n')
    f.close()

if __name__ == '__main__':
    main()
