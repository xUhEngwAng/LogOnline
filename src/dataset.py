import logging
from bisect import bisect_right
from torch.utils.data import Dataset

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class LogDataset(Dataset):
    def __init__(self, sessions_dict, window_size, step_size, padding_idx):
        sessions = []
        windows_cnt_acc = [0]
        
        for key, session in sessions_dict.items():
            session['key'] = key
            session_length = len(session['eventids'])
            
            if session_length == 1:
                continue
            
            windows_cnt = int(max(0, (session_length-window_size))/step_size) + 1
            windows_cnt_acc.append(windows_cnt_acc[-1] + windows_cnt)
            sessions.append(session)
        
        self.padding_idx = padding_idx
        self.sessions = sessions
        self.step_size = step_size
        self.window_size = window_size
        self.windows_cnt_acc = windows_cnt_acc
        
    def __len__(self):
        return self.windows_cnt_acc[-1]
    
    @staticmethod
    def pad(seq, length, padding_idx):
        if len(seq) < length:
            if isinstance(seq[0], int):
                ret = [padding_idx] * (length - len(seq))
                ret.extend(seq)
            elif isinstance(seq[0], float):
                ret = [float(padding_idx)] * (length - len(seq))
                ret.extend(seq)
            elif isinstance(seq[0], str):
                ret = ['<PADDING>'] * (length - len(seq))
                ret.extend(seq)
            elif isinstance(seq[0], list):
                ret = [[]] * (length - len(seq))
                ret.extend(seq)
            else:
                logger.error(f'Fatal error, unrecognized data type for {seq} to pad.')
                exit(0)
            return ret
        return seq
    
    def __getitem__(self, idx):
        session_idx = bisect_right(self.windows_cnt_acc, idx) - 1
        offset = idx - self.windows_cnt_acc[session_idx]
        session = self.sessions[session_idx]
        start = self.step_size * offset
        end = start + self.window_size + 1
        
        session_key = session['key']
        time_elapsed = self.pad(session['time_elapsed'][start: end], self.window_size+1, -1)
        components = self.pad(session['components'][start: end], self.window_size+1, self.padding_idx)
        templates = self.pad(session['templates'][start: end], self.window_size+1, self.padding_idx)
        event_ids = self.pad(session['eventids'][start: end], self.window_size+1, self.padding_idx)
        levels = self.pad(session['levels'][start: end], self.window_size+1, self.padding_idx)
        labels = session['labels'][start: end]
        
        # whether the next log is marked as anomaly
        anomaly = labels[-1]
        next_log = {'eventid': event_ids[-1], 'template': templates[-1]}
        
        return {'session_key': session_key,
                'templates': templates[:-1],
                'eventids': event_ids[:-1],
                'components': components,
                'levels': levels,
                'time_elapsed': time_elapsed,
                'next': next_log,
                'anomaly': anomaly}
