import os

from bs4 import BeautifulSoup
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.snowball import GermanStemmer, EnglishStemmer
from typing import Union, List, Dict, Tuple

class Spider(object):
    """Spider
    """
    
    def __init__(self, corpus:str='data', file_type:str='txt',
        language:str='german') -> None:
        """Constructor

        Args:
            corpus (str, optional): Path to folder which contains the desired
                files for the initial corpus. Defaults to 'data'.
            file_type (str, optional): The type of files to include. Can be
                either 'txt' or 'html'. Other types may result in errors.
                Defaults to 'txt'.
            language (str, optional): Language for creating the tokens. Can be
                either 'german' or 'english'. Defaults to 'german'.
        """           
        self.corpus = corpus
        self.file_type = file_type
        self.content = {}
        self.term_index = {}
        
        self.stopwords=stopwords.words(language)
        self.language=language
        if language == 'german':
            self.stemmer=GermanStemmer()
        elif language == 'english':
            self.stemmer=EnglishStemmer()
        else:
            Exception(f'Unknown language {language}')
        
    def crawl(self, limit:Union[None, int]=None) -> None:
        """Crawl all files

        Crawls all files inside the corpus folder which contain a specific file
        type.

        Args:
            limit (Union[None, int], optional): Number of files to crawl. If
                None all files will be crawled. Defaults to None.
        """
        counter = 0
        for root, dirs, files in os.walk(self.corpus):
            for file in files:
                if file.endswith(f'.{self.file_type}'):

                    file_path = os.path.join(root, file)
                    
                    self.content[file_path] = self.parse(file_path)

                    counter += 1
                    if limit is not None and counter >= limit:
                        return 

    def parse(self, file_path:str) -> str:
        """Parse

        Extracts text from the file. If the file type is 'html', only content of
        paragraphs will be used.

        Args:
            file_path (str): Path to the file.

        Returns:
            str: Text of the file.
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = f.read()
            
            if self.file_type == 'html':
                parser = BeautifulSoup(data,'html.parser')
                paragraphs = parser.find_all('p')   
                content=''
                for p in paragraphs:
                    text=p.text
                    if text is not None and len(text)>0:
                        content+=text             
            else:
                content = data

            return content
        except Exception as e:       
            print(f'error during parsing with {file_path} -> {e}')

    def generate_term_index(self,
        additional_terms:Union[None, List[str]]=None) -> None:
        """Generate Term Index

        Generates the term index for all files of the initial corpus and adds
        `additional_terms` to the corpus. These terms will have a frequency of 1
        if they are not found inside the initial corpus.

        Args:
            additional_terms (Union[None, list[str]], optional): Terms to add
                to the term index. Defaults to None.
        """
        self.clear_index()

        for _, content in self.content.items():
            self.add_new_token(content, count_value=1)
               
        if additional_terms is not None:
            for term in additional_terms:
                self.add_new_token(term, count_value=0)

    def add_new_token(self, content:str, count_value:int=1) -> None:
        """Add New Token

        Extracts tokens from the `content` and adds the frequency to the
        `term_index`. If a token is new to the `term_index`, it will receive a
        frequency of 1, otherwise the frequency is increased by `count_value`.

        Args:
            content (str): Text containing the terms to add.
            count_value (int, optional): Value used to increase the frequency
                of a term. Should be zero if the terms are not part of the
                initial corpus to prevent increasing the frequency of
                additional terms. Defaults to 1.
        """
        tokens = word_tokenize(content, language=self.language)
        for token in tokens:
            # Ignore paranthesis, commas, points etc.
            if len(token)>1 and token not in self.stopwords:
                token=self.stemmer.stem(token)
                if token not in self.term_index:                
                    self.term_index[token]=1
                else:
                    self.term_index[token]+=count_value

    def clear_index(self) -> None:
        """Clear Term Index
        """
        self.term_index = {}

    def get_index_dictionary(self, sort:bool=False) -> Dict[str, int]:
        """Retrieve Term Index as Dictionary

        Args:
            sort (bool, optional): Whether to sort the dictionary based on the
                frequency (high to low). Defaults to False.

        Returns:
            dict[str, int]: Term index as dictionary.
        """
        if sort:
            return {k: v for k, v in sorted(self.term_index.items(),
                key=lambda x: x[1], reverse=True)}
        return self.term_index

    def get_index_list(self, sort:bool=False) -> List[Tuple[int, str]]:
        """Retrieve Term Index as List of Tuple

        Args:
            sort (bool, optional): Whether to sort the list based on the
                frequency (high to low). Defaults to False.

        Returns:
            list[tuple[int, str]]: Term index as list of tuple.
        """
        term_index = []
        for key, value in self.get_index_dictionary(sort=sort).items():
            term_index.append((value, key))
        return term_index
