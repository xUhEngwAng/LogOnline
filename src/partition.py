import logging
import numpy as np

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def partition(parsed_log_df, 
              partition_method,
              train_ratio,
              filter_abnormal,
              session_size):
    if filter_abnormal:
        logger.info('filter_abnormal enabled when creating training data.')
        
    if partition_method == 'session':
        return partitionBySession(parsed_log_df, train_ratio, filter_abnormal)
    else: # partition by timestamp
        return partitionByOrder(parsed_log_df, session_size, train_ratio, filter_abnormal)
    
def partitionByOrder(parsed_log_df,
                     session_size,
                     train_ratio,
                     filter_abnormal):
    num_sessions = len(parsed_log_df) / session_size
    num_train_sessions = int(num_sessions * train_ratio)
    sessions_cnt = 0
    
    session_train, session_test = {}, {}
    eventid2ind = {}
    
    for start in range(0, len(parsed_log_df), session_size):
        end = start + session_size
        labels = parsed_log_df['Anomaly'][start: end].tolist()
        
        if filter_abnormal and True in labels:
            continue
            
        templates = parsed_log_df['Templates'][start: end].tolist()
        event_ids = parsed_log_df['EventId'][start: end].tolist()
        
        for event_id in event_ids:
            eventid2ind.setdefault(event_id, len(eventid2ind))
         
        event_ids = [eventid2ind[event_id] for event_id in event_ids]
        session_train[sessions_cnt] = {'templates': templates, 'eventids': event_ids, 'labels': labels}
        
        sessions_cnt += 1
        if sessions_cnt == num_train_sessions:
            break
            
    num_training_events = len(eventid2ind)
    eventid2ind['<padding>'] = num_training_events
    
    for start in range(start+session_size, len(parsed_log_df), session_size):
        end = start + session_size
        
        labels = parsed_log_df['Anomaly'][start: end].tolist()
        templates = parsed_log_df['Templates'][start: end].tolist()
        event_ids = parsed_log_df['EventId'][start: end].tolist()
        
        for event_id in event_ids:
            eventid2ind.setdefault(event_id, len(eventid2ind))
         
        event_ids = [eventid2ind[event_id] for event_id in event_ids]
        session_test[sessions_cnt] = {'templates': templates, 'eventids': event_ids, 'labels': labels}
        sessions_cnt += 1
        
    # Write back converted event_ids to original Dataframe
    parsed_log_df['EventId'] = parsed_log_df['EventId'].apply(lambda event_id: eventid2ind.get(event_id, event_id))
    
    logger.info(f'partitionByOrder done, {sessions_cnt} sessions are generated.')
    logger.info(f'Sequential Partitioning done. {len(eventid2ind)} event ids are identified.') 
    logger.info(f'Number of training and testing sessions are {len(session_train)} and {len(session_test)}')
    return session_train, session_test, num_training_events
    
def __partitionByOrder(parsed_log_df,
                     session_size,
                     train_ratio,
                     filter_abnormal):
    training_size = int(len(parsed_log_df) * train_ratio)
    templates = parsed_log_df['Templates'].tolist()
    event_ids = parsed_log_df['EventId'].tolist()
    labels = parsed_log_df['Anomaly'].tolist()
    
    # Transform EventId to unified intergers, with EventIds in training data going first
    eventid2ind = {}
    
    for event_id in event_ids[:training_size]:
        eventid2ind.setdefault(event_id, len(eventid2ind))
        
    num_training_events = len(eventid2ind)
    eventid2ind['<padding>'] = num_training_events
    
    for event_id in event_ids[training_size:]:
        eventid2ind.setdefault(event_id, len(eventid2ind))
                               
    event_ids = [eventid2ind[event_id] for event_id in event_ids]
     # Write back converted event_ids to original DataFrame
    parsed_log_df['EventId'] = event_ids
 
    # Construct training and testing data
    session_train = {'all': {'templates': templates[:training_size],
                             'eventids': event_ids[:training_size],
                             'labels': labels[:training_size]}}
    session_test  = {'all': {'templates': templates[training_size:],
                             'eventids': event_ids[training_size:],
                             'labels': labels[training_size:]}}
    logger.info(f'Sequential partitioning done. {len(eventid2ind)} event ids are identified.')
    return session_train, session_test, num_training_events

def partitionBySession(parsed_log_df, 
                       train_ratio, 
                       filter_abnormal):
    session_train, session_test = {}, {}
    groups = parsed_log_df.groupby(by='Session')
    eventid2ind = {}
    test_groups = []
    
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
        
        for event_id in event_ids:
            eventid2ind.setdefault(event_id, len(eventid2ind))
         
        event_ids = [eventid2ind[event_id] for event_id in event_ids]
        session = {'templates': templates, 'eventids': event_ids, 'labels': labels}
        session_train[group_name] = session
        
    num_training_events = len(eventid2ind)
    eventid2ind['<padding>'] = num_training_events
    
    for group_name in test_groups:
        group = groups.get_group(group_name)
        
        templates = group['Templates'].tolist()
        event_ids = group['EventId'].tolist()
        labels = group['Anomaly'].tolist()
        
        for event_id in event_ids:
            eventid2ind.setdefault(event_id, len(eventid2ind))
         
        event_ids = [eventid2ind[event_id] for event_id in event_ids]
        session = {'templates': templates, 'eventids': event_ids, 'labels': labels}
        session_test[group_name] = session
            
    # Write back converted event_ids to original Dataframe
    parsed_log_df['EventId'] = parsed_log_df['EventId'].apply(lambda event_id: eventid2ind.get(event_id, event_id))
    
    logger.info(f'partitionBySession done, {len(groups)} sessions are generated.')
    logger.info(f'Session Partitioning done. {len(eventid2ind)} event ids are identified.') 
    logger.info(f'Number of training and testing sessions are {len(session_train)} and {len(session_test)}')
    return session_train, session_test, num_training_events
    