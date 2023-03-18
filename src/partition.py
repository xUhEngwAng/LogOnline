import logging
import numpy as np
import statistics

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def partition(parsed_log_df, 
              partition_method,
              train_ratio,
              filter_abnormal,
              session_size,
              shuffle):
    if filter_abnormal:
        logger.info('filter_abnormal enabled when creating training data.')
        
    if partition_method == 'session':
        return partitionBySession(parsed_log_df, train_ratio, filter_abnormal)
    else: # partition by timestamp
        if shuffle:
            return partitionByOrderShuffle(parsed_log_df, session_size, train_ratio, filter_abnormal)
        return partitionByOrder(parsed_log_df, session_size, train_ratio, filter_abnormal)
    
def partitionByOrder(parsed_log_df,
                     session_size,
                     train_ratio,
                     filter_abnormal):
    num_sessions = len(parsed_log_df) / session_size
    num_train_sessions = int(num_sessions * train_ratio)
    sessions_cnt = 0
    time_elapsed_training = []
    
    session_train, session_test = {}, {}
    component2ind = {}
    eventid2ind = {}
    level2ind = {}
    
    for start in range(0, len(parsed_log_df), session_size):
        end = start + session_size
        labels = parsed_log_df['Anomaly'][start: end].tolist()
        
        if filter_abnormal and True in labels:
            continue
            
        components = parsed_log_df['Component'][start: end].tolist()
        event_ids = parsed_log_df['EventId'][start: end].tolist()
        levels = parsed_log_df['Level'][start: end].tolist()
        templates = parsed_log_df['Templates'][start: end].tolist()
        time_elapsed = parsed_log_df['TimeElapsed'][start: end].tolist()
        
        time_elapsed_training.extend(time_elapsed)
            
        for ind in range(len(event_ids)):
            component2ind.setdefault(components[ind], len(component2ind))
            eventid2ind.setdefault(event_ids[ind], len(eventid2ind))
            level2ind.setdefault(levels[ind], len(level2ind))
         
        components = [component2ind[component] for component in components]
        event_ids = [eventid2ind[event_id] for event_id in event_ids]
        levels = [level2ind[level] for level in levels]
        
        session_train[sessions_cnt] = {'components': components,
                                       'eventids': event_ids,
                                       'labels': labels, 
                                       'levels': levels,
                                       'templates': templates,
                                       'time_elapsed': time_elapsed}
        
        sessions_cnt += 1
        if sessions_cnt == num_train_sessions:
            break
          
    mean = statistics.mean(time_elapsed_training)
    stddev = statistics.stdev(time_elapsed_training)
    
    for key, session in session_train.items():
        session['time_elapsed'] = [(entry-mean)/stddev for entry in session['time_elapsed']]
    
    logger.info(f'Training sessions generated, mean = {mean}, stddev = {stddev}.')
    num_components, num_events, num_levels = len(component2ind), len(eventid2ind), len(level2ind)
    eventid2ind['<padding>'] = num_events
    
    for start in range(start+session_size, len(parsed_log_df), session_size):
        end = start + session_size
        
        labels = parsed_log_df['Anomaly'][start: end].tolist()
        components = parsed_log_df['Component'][start: end].tolist()
        event_ids = parsed_log_df['EventId'][start: end].tolist()
        levels = parsed_log_df['Level'][start: end].tolist()
        templates = parsed_log_df['Templates'][start: end].tolist()
        time_elapsed = parsed_log_df['TimeElapsed'][start: end].tolist()
        time_elapsed = [(entry-mean)/stddev for entry in time_elapsed]
        
        for ind in range(len(event_ids)):
            component2ind.setdefault(components[ind], len(component2ind))
            eventid2ind.setdefault(event_ids[ind], len(eventid2ind))
            level2ind.setdefault(levels[ind], len(level2ind))
            
        components = [component2ind[component] for component in components]
        event_ids = [eventid2ind[event_id] for event_id in event_ids]
        levels = [level2ind[level] for level in levels]
        
        session_test[sessions_cnt] = {'components': components,
                                      'eventids': event_ids,
                                      'labels': labels, 
                                      'levels': levels,
                                      'templates': templates,
                                      'time_elapsed': time_elapsed}
        sessions_cnt += 1
        
    # Write back converted event_ids to original Dataframe
    parsed_log_df['EventId'] = parsed_log_df['EventId'].apply(lambda event_id: eventid2ind.get(event_id, event_id))
    
    logger.info(f'partitionByOrder done, {sessions_cnt} sessions are generated.')
    logger.info(f'Sequential Partitioning done. {len(eventid2ind)} event ids are identified.') 
    logger.info(f'Number of training and testing sessions are {len(session_train)} and {len(session_test)}.')
    return session_train, session_test, num_components, num_events, num_levels, level2ind

def partitionByOrderShuffle(parsed_log_df,
                            session_size,
                            train_ratio,
                            filter_abnormal):
    session_train, session_test = {}, {}
    eventid2ind = {}
    test_groups = []
    sessions_cnt = 0
    
    for start in range(0, len(parsed_log_df), session_size):
        end = start + session_size
        templates = parsed_log_df['Templates'][start: end].tolist()
        event_ids = parsed_log_df['EventId'][start: end].tolist()
        labels = parsed_log_df['Anomaly'][start: end].tolist()
        
        if filter_abnormal and True in labels:
            test_groups.append({'templates': templates, 'eventids': event_ids, 'labels': labels})
            continue
        if np.random.random() > train_ratio:
            test_groups.append({'templates': templates, 'eventids': event_ids, 'labels': labels})
            continue
            
        for event_id in event_ids:
            eventid2ind.setdefault(event_id, len(eventid2ind))
         
        event_ids = [eventid2ind[event_id] for event_id in event_ids]
        session_train[sessions_cnt] = {'templates': templates, 'eventids': event_ids, 'labels': labels}
        sessions_cnt += 1
       
    num_training_events = len(eventid2ind)
    eventid2ind['<padding>'] = num_training_events
    
    for test_session in test_groups:
        event_ids = test_session['eventids']
        
        for event_id in event_ids:
            eventid2ind.setdefault(event_id, len(eventid2ind))
         
        test_session['eventids'] = [eventid2ind[event_id] for event_id in event_ids]
        session_test[sessions_cnt] = test_session
        sessions_cnt += 1
        
    # Write back converted event_ids to original Dataframe
    parsed_log_df['EventId'] = parsed_log_df['EventId'].apply(lambda event_id: eventid2ind.get(event_id, event_id))
    
    logger.info(f'partitionByOrderShuffle done, {sessions_cnt} sessions are generated.')
    logger.info(f'Sequential Partitioning done. {len(eventid2ind)} event ids are identified.') 
    logger.info(f'Number of training and testing sessions are {len(session_train)} and {len(session_test)}.')
    return session_train, session_test, num_training_events

def partitionBySession(parsed_log_df, 
                       train_ratio, 
                       filter_abnormal):
    session_train, session_test = {}, {}
    groups = parsed_log_df.groupby(by='Session')
    component2ind, eventid2ind, level2ind = {}, {}, {}
    test_groups = []
    time_elapsed_training = []
    
    for group_name, group in groups:
        labels = group['Anomaly'].tolist()
        if filter_abnormal and True in labels:
            test_groups.append(group_name)
            continue
        
        if np.random.random() > train_ratio:
            test_groups.append(group_name)
            continue
            
        templates = group['Templates'].tolist()
        event_ids = group['EventId'].tolist()
        components = group['Component'].tolist()
        levels = group['Level'].tolist()
        timestamps = group['Timestamp'].tolist()
        
        time_elapsed = [0]
        time_elapsed.extend([timestamps[ind] - timestamps[ind-1] for ind in range(1, len(timestamps))])
        time_elapsed_training.extend(time_elapsed)
        
        for ind in range(len(event_ids)):
            component2ind.setdefault(components[ind], len(component2ind))
            eventid2ind.setdefault(event_ids[ind], len(eventid2ind))
            level2ind.setdefault(levels[ind], len(level2ind))
         
        components = [component2ind[component] for component in components]
        event_ids = [eventid2ind[event_id] for event_id in event_ids]
        levels = [level2ind[level] for level in levels]
        
        session = {'components': components,
                   'eventids': event_ids,
                   'labels': labels, 
                   'levels': levels,
                   'templates': templates,
                   'time_elapsed': time_elapsed}
        session_train[group_name] = session
        
    mean = statistics.mean(time_elapsed_training)
    stddev = statistics.stdev(time_elapsed_training)
    
    for key, session in session_train.items():
        session['time_elapsed'] = [(entry-mean)/stddev for entry in session['time_elapsed']]
    
    logger.info(f'Training sessions generated, mean = {mean}, stddev = {stddev}.')    
    num_components, num_events, num_levels = len(component2ind), len(eventid2ind), len(level2ind)
    eventid2ind['<padding>'] = num_events
    
    for group_name in test_groups:
        group = groups.get_group(group_name)
        
        templates = group['Templates'].tolist()
        event_ids = group['EventId'].tolist()
        components = group['Component'].tolist()
        labels = group['Anomaly'].tolist()
        levels = group['Level'].tolist()
        timestamps = group['Timestamp'].tolist()
        
        time_elapsed = [0]
        time_elapsed.extend([timestamps[ind] - timestamps[ind-1] for ind in range(1, len(timestamps))])
        time_elapsed = [(entry-mean)/stddev for entry in time_elapsed]
        
        for ind in range(len(event_ids)):
            component2ind.setdefault(components[ind], len(component2ind))
            eventid2ind.setdefault(event_ids[ind], len(eventid2ind))
            level2ind.setdefault(levels[ind], len(level2ind))
         
        components = [component2ind[component] for component in components]
        event_ids = [eventid2ind[event_id] for event_id in event_ids]
        levels = [level2ind[level] for level in levels]
        
        session = {'components': components,
                   'eventids': event_ids,
                   'labels': labels, 
                   'levels': levels,
                   'templates': templates,
                   'time_elapsed': time_elapsed}
        session_test[group_name] = session
            
    # Write back converted event_ids to original Dataframe
    parsed_log_df['EventId'] = parsed_log_df['EventId'].apply(lambda event_id: eventid2ind.get(event_id, event_id))
    
    logger.info(f'partitionBySession done, {len(groups)} sessions are generated.')
    logger.info(f'Session Partitioning done. {len(eventid2ind)} event ids are identified.') 
    logger.info(f'Number of training and testing sessions are {len(session_train)} and {len(session_test)}')
    return session_train, session_test, num_components, num_events, num_levels, level2ind
    