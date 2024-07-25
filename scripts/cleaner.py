import re
import pandas as pd

class DataCleaner:
    def __init__(self, data = None):
        self.data = data

    def normalize_char_level_missmatch(self, input_token):
        rep=re.sub('[ሃኅኃሐሓኻ]','ሀ',input_token)
        rep=re.sub('[ሑኁዅ]','ሁ',rep)
        rep=re.sub('[ኂሒኺ]','ሂ',rep)
        rep=re.sub('[ኌሔዄ]','ሄ',rep)
        rep=re.sub('[ሕኅ]','ህ',rep)
        rep=re.sub('[ኆሖኾ]','ሆ',rep)
        rep=re.sub('[ሠ]','ሰ',rep)
        rep=re.sub('[ሡ]','ሱ',rep)
        rep=re.sub('[ሢ]','ሲ',rep)
        rep=re.sub('[ሣ]','ሳ',rep)
        rep=re.sub('[ሤ]','ሴ',rep)
        rep=re.sub('[ሥ]','ስ',rep)
        rep=re.sub('[ሦ]','ሶ',rep)
        rep=re.sub('[ዓኣዐ]','አ',rep)
        rep=re.sub('[ዑ]','ኡ',rep)
        rep=re.sub('[ዒ]','ኢ',rep)
        rep=re.sub('[ዔ]','ኤ',rep)
        rep=re.sub('[ዕ]','እ',rep)
        rep=re.sub('[ዖ]','ኦ',rep)
        rep=re.sub('[ጸ]','ፀ',rep)
        rep=re.sub('[ጹ]','ፁ',rep)
        rep=re.sub('[ጺ]','ፂ',rep)
        rep=re.sub('[ጻ]','ፃ',rep)
        rep=re.sub('[ጼ]','ፄ',rep)
        rep=re.sub('[ጽ]','ፅ',rep)
        rep=re.sub('[ጾ]','ፆ',rep)
        
        #Normalizing words with Labialized Amharic characters such as በልቱዋል or  በልቱአል to  በልቷል  
        rep=re.sub('(ሉ[ዋአ])','ሏ',rep)
        rep=re.sub('(ሙ[ዋአ])','ሟ',rep)
        rep=re.sub('(ቱ[ዋአ])','ቷ',rep)
        rep=re.sub('(ሩ[ዋአ])','ሯ',rep)
        rep=re.sub('(ሱ[ዋአ])','ሷ',rep)
        rep=re.sub('(ሹ[ዋአ])','ሿ',rep)
        rep=re.sub('(ቁ[ዋአ])','ቋ',rep)
        rep=re.sub('(ቡ[ዋአ])','ቧ',rep)
        rep=re.sub('(ቹ[ዋአ])','ቿ',rep)
        rep=re.sub('(ሁ[ዋአ])','ኋ',rep)
        rep=re.sub('(ኑ[ዋአ])','ኗ',rep)
        rep=re.sub('(ኙ[ዋአ])','ኟ',rep)
        rep=re.sub('(ኩ[ዋአ])','ኳ',rep)
        rep=re.sub('(ዙ[ዋአ])','ዟ',rep)
        rep=re.sub('(ጉ[ዋአ])','ጓ',rep)
        rep=re.sub('(ደ[ዋአ])','ዷ',rep)
        rep=re.sub('(ጡ[ዋአ])','ጧ',rep)
        rep=re.sub('(ጩ[ዋአ])','ጯ',rep)
        rep=re.sub('(ጹ[ዋአ])','ጿ',rep)
        rep=re.sub('(ፉ[ዋአ])','ፏ',rep)
        rep=re.sub('[ቊ]','ቁ',rep) #ቁ can be written as ቊ
        rep=re.sub('[ኵ]','ኩ',rep) #ኩ can be also written as ኵ  
        return rep

    def remove_punc_and_special_chars(self,text):
        try: 
            # pattern = r'[\!\@\#\$\%\^\«\»\&\*\(\)\…\[\]\{\}\;\“\”\›\’\‘\"\'\:\,\.\‹\/\<\>\?\\\\|\`\´\~\-\=\+\፡\።\፤\;\፦\፥\፧\፨\፠\፣0-9]|1234567890'
            # normalized_text = re.sub(pattern, '', text)
            normalized_text = re.sub('[\!\@\#\$\%\^\«\»\&\*\(\)\…\[\]\{\}\;\“\”\›\’\‘\"\'\:\,\.\‹\/\<\>\?\\\\|\`\´\~\-\=\+\፡\።\፤\;\፦\፥\፧\፨\፠\፣]', '',text) 
            return normalized_text
        except Exception as e:
            raise ValueError(
                "An error occurred while removing punctuation and special characters from the input text. Exception: {}".format(e)) from e
    

    #remove all ascii characters and Arabic and Amharic numbers
    def remove_ascii_and_numbers(self,text_input):
        try:
            rm_num_and_ascii=re.sub('[A-Za-z0-9]','',text_input)
            return re.sub('[\'\u1369-\u137C\']+','',rm_num_and_ascii)
        except Exception as e:
            raise ValueError(
                "An error occurred while removing punctuation and special characters from the input text. Exception: {}".format(e)) from e
        
    def export_to_csv(self, df: pd.DataFrame, file_name: str):
        try:
            df.to_csv(file_name, index=False)
        except Exception as e:
            raise ValueError(
                "An error occurred while removing punctuation and special characters from the input text. Exception: {}".format(e)) from e
    