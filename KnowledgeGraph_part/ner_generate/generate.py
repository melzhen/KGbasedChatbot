from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline

if __name__ == "__main__":
        sentences=""
        path = "medaa"
        with open(path, 'r') as f:
                for x in f:
                        sentences+=x
        sent_ls=sentences.split('===')
        tokenizer = AutoTokenizer.from_pretrained("dslim/bert-base-NER")
        model = AutoModelForTokenClassification.from_pretrained("dslim/bert-base-NER")
        nlp = pipeline("ner", model=model, tokenizer=tokenizer)
        output=[]
        for i, sent in enumerate(sent_ls):
                ner_result = nlp(sent)
                for x in ner_result:
                        output.append(x['word']+','+x['entity']+','+str(i)+'\n')
        with open('ner_aa.csv','w') as f:
                f.writelines(output)
        
        path = "medab"
        sentences=""
        with open(path, 'r') as f:
                for x in f:
                        sentences+=x
        sent_ls=sentences.split('===')
        tokenizer = AutoTokenizer.from_pretrained("dslim/bert-base-NER")
        model = AutoModelForTokenClassification.from_pretrained("dslim/bert-base-NER")
        nlp = pipeline("ner", model=model, tokenizer=tokenizer)
        output=[]
        for i, sent in enumerate(sent_ls):
                ner_result = nlp(sent)
                for x in ner_result:
                        output.append(x['word']+','+x['entity']+','+str(i+15313)+'\n')
        with open('ner_ab.csv','w') as f:
                f.writelines(output)
            
        path = "medac"
        sentences=""
        with open(path, 'r') as f:
                for x in f:
                        sentences+=x
        sent_ls=sentences.split('===')
        tokenizer = AutoTokenizer.from_pretrained("dslim/bert-base-NER")
        model = AutoModelForTokenClassification.from_pretrained("dslim/bert-base-NER")
        nlp = pipeline("ner", model=model, tokenizer=tokenizer)
        output=[]
        for i, sent in enumerate(sent_ls):
                ner_result = nlp(sent)
                for x in ner_result:
                        output.append(x['word']+','+x['entity']+','+str(i+15313+15312)+'\n')
        with open('ner_ac.csv','w') as f:
                f.writelines(output)
        
        path = "medad"
        sentences=""
        with open(path, 'r') as f:
                for x in f:
                        sentences+=x
        sent_ls=sentences.split('===')
        tokenizer = AutoTokenizer.from_pretrained("dslim/bert-base-NER")
        model = AutoModelForTokenClassification.from_pretrained("dslim/bert-base-NER")
        nlp = pipeline("ner", model=model, tokenizer=tokenizer)
        output=[]
        for i, sent in enumerate(sent_ls):
                ner_result = nlp(sent)
                for x in ner_result:
                        output.append(x['word']+','+x['entity']+','+str(i+15313+15312+15314)+'\n')
        with open('ner_ad.csv','w') as f:
                f.writelines(output)
