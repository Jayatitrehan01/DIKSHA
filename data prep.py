'''
Download a working subset of the synthetic discharge notes and save:
data/notes.json ->[{}]
data/qa_pairs.json ->[{}]

These are synthetic discharge-summary-style dcouments, so there is not patient privacy concern.
The qa pairs are labeled by NLP task(summarization,paraphasing,relation extraction and so on) rather than patient-style questions, so they are used here as retrieval test material. For the strongest evaluation, also hand-write a small set of realistic pateint questions

'''
import json
import os

from datasets import load_dataset

n_notes=300 #no. of unique discharge notes to keep for the demo
max_qa=300 #cap on the no. of QA pairs kept for evaluation

def main():
    os.makedirs('data',exist_ok=True)

    # Streaming avoids downloading the entire (large) dataset to disk.

    ds= load_dataset(
        'notes.json',
        split='train',
        streaming=True,
    )

    notes={}          #patient_id -> notes text (deduplicated)
    qa_pairs = [ ]    # evaluation material

    for row in ds:
        p_id=row['patient_id']

        if p_id not in notes and len(notes)<n_notes:
            notes[p_id]=row['note']

        if p_id in notes and len(qa_pairs)< max_qa:
            qa_pairs.append({
                'patient_id':p_id,
                'question': row['question'],
                'answer':row['answer'],
                'task':row['task']
            })
    
        if len(notes)>=n_notes and len(qa_pairs)>=max_qa:
            break
        
        with open('data/notes.json','w') as f:
            json.dump([{'patient_id':k,'notes': v} for k, v in notes.items()],indent=2)
            
        with open('data/qa_pairs.json', 'w')as f:
            json.dump(qa_pairs,f,indent=2)
            
        print(f'Saved {len(notes)} NOTES  -> data/notes.json')
        print(f'Saved {len(qa_pairs)}QA PAIRS -> data/qa_pairs.json')


if __name__ == '__main__':
    main()