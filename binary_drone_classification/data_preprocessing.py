import sys
import platform
current_os = platform.system()
if current_os == 'Darwin':  # macOS
    sys.path.append('/Users/simonmyhre/workdir/gitdir/skyline')
elif current_os == 'Linux':  # Linux
    sys.path.append('/cluster/datastore/simonmy/skyline')
from cirrus import Data

import os
from dotenv import load_dotenv
import os
load_dotenv()


if __name__ == "__main__":
    data = Data(os.getenv("DATA"))
    data.window_it(1)
    data.split_it(train_percent=70, test_percent=20, validation_percent=10)
    data.label_to_class_map_it({
        'drone': ['normal_drone', 'racing_drone', 'normal_fixedwing', 'petrol_fixedwing'],
        'non-drone': ['no_class']
    })
    data.sample_rate_it(44100)
    data.augment_it(['low_pass'])
    data.audio_format_it('stft')
    data.file_type_it('npy')
    data.describe_it()
    data.run_it(output_path_to_data="cache/data")
