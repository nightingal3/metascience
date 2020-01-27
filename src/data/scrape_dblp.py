import csv
import os
import sys
import time

from bs4 import BeautifulSoup
import requests
from selenium import webdriver
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.wait import WebDriverWait


url = "https://dblp.uni-trier.de/pers/hd"
link_xpath = "//div[@class='box'][img[@title='Journal Articles' or @title='Conference and Workshop Papers']]/following-sibling::nav[@class='publ']/ul/*[1]/div[@class='head']"
timeout = 8
scroll_pause_time = 0.25

def scrape(affix: str, filename: str, get_abstracts: bool = False) -> None:
    req = requests.get(url + affix)
    data = req.text
    if get_abstracts:
        driver = webdriver.Chrome(executable_path="./chromedriver.exe")
        driver.get(url + affix)

    driver.set_page_load_timeout(timeout)

    soup = BeautifulSoup(data, features="html.parser")
    entries = [e.find("cite", {"class": "data"}) for e in soup.findAll("li", {"class": ["article", "inproceedings"]})]
    papers = []
    abstracts = []
   
    if get_abstracts:
        # Get the abstracts from a variety of third party websites
        footer = WebDriverWait(driver, timeout).until(
            lambda d: d.find_element_by_id("footer")) # infinite scroll...
        ActionChains(driver).move_to_element(footer).perform()
        header = driver.find_element_by_id("logo")
        ActionChains(driver).move_to_element(header).perform()

        links = WebDriverWait(driver, timeout).until(
            lambda d: d.find_elements_by_xpath(link_xpath))
        for i in range(len(links)):
            try:
                href = links[i].find_element_by_xpath(".//a").get_attribute("href")
            except:
                abstracts.append("n/a")
                continue


            if href[-3:] == ".ps" or href[-3:] == ".gz":
                abstracts.append("n/a")
                continue

            driver.get(href)
            #time.sleep(timeout)
            try:
                if (len(driver.find_elements_by_css_selector(".site-logo__springer")) != 0):
                    ab = driver.find_element_by_xpath("//section[@class='Abstract']")
                elif (len(driver.find_elements_by_css_selector("#xplore-header")) != 0):
                    ab = driver.find_element_by_xpath("//div[@class='abstract-text row']")
                elif (len(driver.find_elements_by_xpath("//img[contains(@alt, 'ACM DL')]")) != 0):
                    ab = driver.find_element_by_xpath("//div[@id='abstract-body']")
                elif (len(driver.find_elements_by_xpath("//img[contains(@alt, 'IEEE Computer Society Digital Library')]")) != 0):
                    ab = driver.find_element_by_xpath("//div[contains(@class, 'article-content')]")
                elif (len(driver.find_elements_by_css_selector(".jmlr")) != 0):
                    ab = driver.find_element_by_xpath("//div[@id='abstract']")
                elif (len(driver.find_elements_by_css_selector("#cu-identity")) != 0):
                    ab = driver.find_element_by_xpath("//blockquote")
                elif (len(driver.find_elements_by_xpath("//div[@class='abstractSection abstractInFull']")) != 0):
                    ab - driver.find_element_by_xpath("//div[@class='abstractSection abstractInFull']/p")
                elif ("papers.nips.cc" in driver.current_url):
                    ab = driver.find_element_by_xpath("//p[@class='abstract']")
                elif ("aaai.org/ojs/index.php/aimagazine" in driver.current_url):
                    ab = driver.find_element_by_css_selector(".abstract")
                elif ("aaai.org/Library" in driver.current_url):
                    ab = driver.find_element_by_xpath("//p[@class='left']/following-sibling::p[1]")
                elif ("sciencedirect.com/science/article" in driver.current_url):
                    ab = driver.find_element_by_css_selector("#abstracts")
                elif ("link.springer.com/article" in driver.current_url):
                    ab = driver.find_element_by_id("c-article-section__content")
                elif ("sciencedirect.com" in driver.current_url):
                    ab = driver.find_element_by_css_selector(".abstract")
                elif ("openreview.net" in driver.current_url):
                    ab = driver.find_element_by_css_selector(".note_content_value")
                elif ("https://aaai.org/" in driver.current_url):
                    ab = driver.find_element_by_xpath("//div[@class='item abstract']")
                elif ("aclweb.org" in driver.current_url):
                    ab = driver.find_element_by_xpath("//div[@class='card-body acl-abstract']")
                elif ("dl.acm.org" in driver.current_url):
                    ab = driver.find_element_by_xpath("//div[@class='article__section article__abstract hlFld-Abstract']")
                elif ("dblp.uni-trier.de" in driver.current_url): # dead link
                    abstracts.append("n/a")
                    links = WebDriverWait(driver, timeout).until(
                        lambda d: d.find_elements_by_xpath(link_xpath))
                    continue
                else:
                    abstracts.append("n/a")
                    driver.back()
                    #time.sleep(timeout)
                    links = WebDriverWait(driver, timeout).until(
                        lambda d: d.find_elements_by_xpath(link_xpath))
                    continue
                
                abstracts.append(ab.text)
                driver.back()
                #time.sleep(timeout)
                links = WebDriverWait(driver, timeout).until(
                        lambda d: d.find_elements_by_xpath(link_xpath))
            except:
                abstracts.append("n/a")
                driver.back()
                #time.sleep(timeout)
                links = WebDriverWait(driver, timeout).until(
                        lambda d: d.find_elements_by_xpath(link_xpath))
                continue
        
    if get_abstracts:
        assert len(links) == len(abstracts)
        
    for i, e in enumerate(entries):
        title = e.find("span", {"class": "title"}).text
        year = e.find("span", {"itemprop": "datePublished"}).text
        if year == []:
            year = driver.find_element_by_xpath(f"//span[contains(string(), {title})]/ancestor:li[contains(@class, 'entry')]/preceding-sibling::li[@class='year']").text
        if title == [] or year == []:
            continue
        if get_abstracts:
            papers.append((year, title, abstracts[i]))
        else:
            papers.append((year, title))
    with open(filename, "w") as out:
        csv_out = csv.writer(out)
        for row in papers:
            csv_out.writerow(row)

def scrape_names(filename: str, out_path: str, get_abstracts=False) -> None:
    with open(filename, "r") as name_file:
        reader = csv.reader(name_file)
        for line in reader:
            split_name = line[0].split(" ")
            split_name = [x.replace(".", "=") for x in split_name]
            given_name = split_name[0] if len(split_name) == 2 else f"{split_name[0]}_{split_name[1]}"
            affix = f"/{split_name[-1][0].lower()}/{split_name[-1]}:{given_name}"
            filename = f"{out_path}/{given_name}-{split_name[-1]}.csv"
            if os.path.exists(filename):
                continue

            scrape(affix, filename, get_abstracts)

if __name__ == "__main__":
    #scrape_names("data/turing_winners/turing_winners.txt", "data/turing_winners/abstracts", get_abstracts=True)
    scrape("/b/Brooks_Jr=:Frederick_P=", "Frederick-P-Brooks-Jr-abstracts.csv", get_abstracts=True)