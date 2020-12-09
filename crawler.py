from bs4 import BeautifulSoup
import requests
import os
import re
import sys
from pprint import pprint
import time
import csv
import pandas as pd

def get_soup(url):
    r=requests.get(url).text
    return BeautifulSoup(r,'html.parser')

def get_basic_page(book_id):
    url = url_base.format(book_id)
    return get_soup(url)

def _parse_title(page):
    try:
        title=page.select('div[class=book_info] h2 a')
        return title[0].text.replace("\xa0",' ')
    except:
        return ''

def _parse_author(page):
    try:
        author=page.select('div[class=book_info_inner] div a')
        return author[1].text
    except:
        return ''

def _parse_score(page):
    try:
        score=page.select('div[class=txt_desc] a strong')
        return score[0].text
    except:
        return ''

def _parse_num_reviewers(page):
    try:
        score=page.select('div[class=txt_desc] a strong')
        return score[1].text
    except:
        return ''

def _parse_genre1(page):
    try:
        genre=genre=page.select('div[class=location] li[class=select]')
        return genre[0].text.replace("\n",'')
    except:
        return ''

def _parse_genre2(page):
    try:
        genre=genre=page.select('div[class=location] li[class=select2]') 
        return genre[0].text.replace("\n",'')
    except:
        return ''

def _parse_genre3(page):
    try:
        genre=genre=page.select('div[class=location] li[class=select3]') 
        return genre[0].text.replace("\n",'')
    except:
        return ''
    
def _parse_book_intro(page):
    try:
        book_intro=page.select('div[id=bookIntroContent] p')
        return book_intro[0].text
    except:
        return ''

def _parse_book_review(page):
    try:
        review=page.select('div[id=pubReviewContent] p')
        return review[0].text
    except:
        return ''
    
def _parse_basic_page(page):
    book={}
    
    book['title']=_parse_title(page)
    book['author']=_parse_author(page)
    book['score']=_parse_score(page)
    book['num_reviewers']=_parse_num_reviewers(page)
    book['main genre']=_parse_genre1(page)
    book['middle genre']=_parse_genre2(page)
    book['subclass']=_parse_genre3(page)
    book['introduction']=_parse_book_intro(page)
    book['review']=_parse_book_review(page)
    
    return book

url_base='https://book.naver.com/bookdb/book_detail.nhn?bid={}'

book_id_list=pd.read_csv('book_id_list(정치&사회).csv',encoding='utf-8')

id_list=book_id_list['book_list'].unique()

book_dic_data=[]

for book_id in id_list:
    url=url_base.format(book_id)
    page=get_soup(url)
    book_dic_data+=[_parse_basic_page(page)]
    time.sleep(1.0)
    
csv_columns=['title','author','score','num_reviewers','main genre',
            'middle genre','subclass','introduction','review']
with open('book_info(정치&사회).csv','w',-1,"utf-8-sig") as csvfile:
    writer=csv.DictWriter(csvfile,fieldnames=csv_columns)
    writer.writeheader()
    for data in book_dic_data:
        writer.writerow(data)

pd.read_csv('book_info(정치&사회).csv',encoding='utf-8')       
        