
import spacy
from nltk import tokenize as nltk_tokenize_sentence
import pickle
import torch
import torch.nn as nn
import re


def filter(token):
    if token[0] == '@':
        return '<at_@>'
    if token[:4] == 'http':
        return '<http>'
    return token.lower()


nlp = spacy.load('en')
sentence = ' amazom kicking there ass. '

tokenized = [filter(tok.text) for tok in nlp.tokenizer(sentence)]

# print(tokenized)


sentence = "Neoliberalism as I and everyone else is talking about it started in the 1970s when people started to see that socialism wasn't working so well so they shifted back towards capitalism. \n\nWikipedia actually has a really good summary of it that covers your misunderstanding over the usage of the terminology. \n\n&gt;**Neoliberalism** or **neo-liberalism**[\\[1\\]](https://en.wikipedia.org/wiki/Neoliberalism#cite_note-1) is the 20th-century resurgence of 19th-century ideas associated with [*laissez-faire*](https://en.wikipedia.org/wiki/Laissez-faire) [economic liberalism](https://en.wikipedia.org/wiki/Economic_liberalism) and [free market](https://en.wikipedia.org/wiki/Free_market) [capitalism](https://en.wikipedia.org/wiki/Capitalism).[\\[2\\]](https://en.wikipedia.org/wiki/Neoliberalism#cite_note-Haymes-2):7[\\[3\\]](https://en.wikipedia.org/wiki/Neoliberalism#cite_note-3) Those ideas include [economic liberalization](https://en.wikipedia.org/wiki/Economic_liberalization) policies such as [privatization](https://en.wikipedia.org/wiki/Privatization), [austerity](https://en.wikipedia.org/wiki/Austerity), [deregulation](https://en.wikipedia.org/wiki/Deregulation), [free trade](https://en.wikipedia.org/wiki/Free_trade)[\\[4\\]](https://en.wikipedia.org/wiki/Neoliberalism#cite_note-4) and reductions in [government spending](https://en.wikipedia.org/wiki/Government_spending) in order to increase the role of the [private sector](https://en.wikipedia.org/wiki/Private_sector) in the [economy](https://en.wikipedia.org/wiki/Economy) and [society](https://en.wikipedia.org/wiki/Society).[\\[12\\]](https://en.wikipedia.org/wiki/Neoliberalism#cite_note-12) These market-based ideas and the policies they inspired constitute a [paradigm shift](https://en.wikipedia.org/wiki/Paradigm_shift) away from the post-war [Keynesian](https://en.wikipedia.org/wiki/Keynesian_economics) consensus which lasted from 1945 to 1980.[\\[13\\]](https://en.wikipedia.org/wiki/Neoliberalism#cite_note-FPIF-2004-13)[\\[14\\]](https://en.wikipedia.org/wiki/Neoliberalism#cite_note-14)  \n&gt;  \n&gt;English-speakers have used the term \"neoliberalism\" since the start of the 20th century with different meanings,[\\[15\\]](https://en.wikipedia.org/wiki/Neoliberalism#cite_note-OxfordNeoliberalism-15) but it became more prevalent in its current meaning in the 1970s and 1980s, used by scholars in a wide variety of [social sciences](https://en.wikipedia.org/wiki/Social_science)[\\[16\\]](https://en.wikipedia.org/wiki/Neoliberalism#cite_note-16)[\\[17\\]](https://en.wikipedia.org/wiki/Neoliberalism#cite_note-Handbookscholarship-17) as well as by critics.[\\[18\\]](https://en.wikipedia.org/wiki/Neoliberalism#cite_note-18)[\\[19\\]](https://en.wikipedia.org/wiki/Neoliberalism#cite_note-Jones2014-19) Modern advocates of free market policies avoid the term \"neoliberal\"[\\[20\\]](https://en.wikipedia.org/wiki/Neoliberalism#cite_note-20) and some scholars have described the term as meaning different things to different people[\\[21\\]](https://en.wikipedia.org/wiki/Neoliberalism#cite_note-21)[\\[22\\]](https://en.wikipedia.org/wiki/Neoliberalism#cite_note-22) as neoliberalism \"mutated\" into geopolitically distinct hybrids as it travelled around the world.[\\[5\\]](https://en.wikipedia.org/wiki/Neoliberalism#cite_note-Handbook2-5) As such, neoliberalism shares many attributes with other concepts that have contested meanings, including [democracy](https://en.wikipedia.org/wiki/Democracy).[\\[23\\]](https://en.wikipedia.org/wiki/Neoliberalism#cite_note-23)  \n&gt;  \n&gt;The definition and usage of the term have changed over time.[\\[6\\]](https://en.wikipedia.org/wiki/Neoliberalism#cite_note-Boas2009-6) As an [economic philosophy](https://en.wikipedia.org/wiki/Economic_philosophy), neoliberalism emerged among European [liberal](https://en.wikipedia.org/wiki/Liberalism) scholars in the 1930s as they attempted to trace a so-called \"third\" or \"middle\" way between the conflicting philosophies of [classical liberalism](https://en.wikipedia.org/wiki/Classical_liberalism) and [socialist planning](https://en.wikipedia.org/wiki/Economic_planning).[\\[24\\]](https://en.wikipedia.org/wiki/Neoliberalism#cite_note-mirowski-24):14\u201315 The impetus for this development arose from a desire to avoid repeating the economic failures of the early 1930s, which neoliberals mostly blamed on the [economic policy](https://en.wikipedia.org/wiki/Economic_policy) of classical liberalism. In the decades that followed, the use of the term \"neoliberal\" tended to refer to theories which diverged from the more *laissez-faire* doctrine of classical liberalism and which promoted instead a [market economy](https://en.wikipedia.org/wiki/Market_economy) under the guidance and rules of a strong state, a model which came to be known as the [social market economy](https://en.wikipedia.org/wiki/Social_market_economy).  \n&gt;  \n&gt;In the 1960s, usage of the term \"neoliberal\" heavily declined. When the term re-appeared in the 1980s in connection with [Augusto Pinochet](https://en.wikipedia.org/wiki/Augusto_Pinochet)'s [economic reforms](https://en.wikipedia.org/wiki/Economic_history_of_Chile#%22Neoliberal%22_reforms_(1973%E2%80%9390)) in [Chile](https://en.wikipedia.org/wiki/Chile), the usage of the term had shifted. It had not only become a term with negative connotations employed principally by critics of market reform, but it also had shifted in meaning from a moderate form of liberalism to a more radical and *laissez-faire* capitalist set of ideas. Scholars now tended to associate it with the theories of [Mont Pelerin Society](https://en.wikipedia.org/wiki/Mont_Pelerin_Society) economists [Friedrich Hayek](https://en.wikipedia.org/wiki/Friedrich_Hayek), [Milton Friedman](https://en.wikipedia.org/wiki/Milton_Friedman), and [James M. Buchanan](https://en.wikipedia.org/wiki/James_M._Buchanan), along with politicians and policy-makers such as [Margaret Thatcher](https://en.wikipedia.org/wiki/Margaret_Thatcher), [Ronald Reagan](https://en.wikipedia.org/wiki/Ronald_Reagan) and [Alan Greenspan](https://en.wikipedia.org/wiki/Alan_Greenspan).[\\[6\\]](https://en.wikipedia.org/wiki/Neoliberalism#cite_note-Boas2009-6)[\\[25\\]](https://en.wikipedia.org/wiki/Neoliberalism#cite_note-25) Once the new meaning of neoliberalism became established as a common usage among Spanish-speaking scholars, it diffused into the English-language study of [political economy](https://en.wikipedia.org/wiki/Political_economy).[\\[6\\]](https://en.wikipedia.org/wiki/Neoliberalism#cite_note-Boas2009-6) By 1994, with the passage of [NAFTA](https://en.wikipedia.org/wiki/NAFTA) and with the [Zapatistas](https://en.wikipedia.org/wiki/Zapatista_Army_of_National_Liberation)' reaction to this development in [Chiapas](https://en.wikipedia.org/wiki/Chiapas), the term entered global circulation.[\\[5\\]](https://en.wikipedia.org/wiki/Neoliberalism#cite_note-Handbook2-5) Scholarship on the phenomenon of neoliberalism has been growing over the last couple of decades.[\\[17\\]](https://en.wikipedia.org/wiki/Neoliberalism#cite_note-Handbookscholarship-17)[\\[26\\]](https://en.wikipedia.org/wiki/Neoliberalism#cite_note-26)\n\n&amp;#x200B;\n\n&amp;#x200B;"
sentence = sentence.replace('](https:', '] (https:')
tokenized = [filter(tok.text) for tok in nlp.tokenizer(sentence)]

# print(len(sentence))
# print(len(tokenized))
# print(tokenized)


def preprocess_text(text):
    return ' '.join(re.findall(r"[\w']+|[.,!?;]", text))


def filter_v2(token):
    if token[0] == '@':
        return '<at_@>'
    if token[:4] == 'http':
        return '<http>'
    return token


def preprocess_text_v2(text):
	text = " ".join(text.split())
	text = text.replace('](https:', '] (https:')
	text = text.replace('](http:', '] (http:')
	text = text.replace('\u2019', "'")
	text = text.replace('&amp;', "&")
	sentence = [filter_v2(tok.text) for tok in nlp.tokenizer(text)]
	sentence = ' '.join(sentence)

	return sentence


text = "a  b c  \n\n  d \n"
print(preprocess_text_v2(text))
print(text)
# exit(0)


text = 'I\'m running Rain, 2B, V. Welch and bringing 9S or Bride/Awakened Maria for my loan unit. Has been much smoother than multiplayer so far, very easy sub-1:30 runs on M3. Rain and 2B are both fully seeded. Rain is using an Icicle Sword, and I try to get a loan unit with ice weapons as well if possible. I control Rain and hit around 500k damage per Lava Floor post-rush.'
print(text)
print(preprocess_text_v2(text))
print('\n\n')

text = '[https://www.reddit.com/r/arduino/search?q=project%20ideas&amp;restrict\\_sr=1] (https://www.reddit.com/r/arduino/search?q=project%20ideas&amp;restrict_sr=1)'
print(text)
print(preprocess_text_v2(text))
print('\n\n')


text = "At face value it's a pretty basic thing, but it's far from it.. On motion detected, tablet wakes up and shows a picture slideshow, and touching it displays the home assistant front end. Simple on Android with Fully kiosks, right? This is on an iPad however.\n\nIt's max iOS version is 10.3.5 which luckily is jailbreakable. So I jailbroke it and installed an SSH tweak, and a tweak called Activator, which let's me control stuff remotely. Then I generated an SSH key pair between my hassio install and  the iPad which lets hassio ssh into it without password. I set up a few command line switches which send an ssh command triggering activator to do different things. In this case, if motion detected, tell the iPad (Activator) that both volume buttons were pressed simultaneously - which acts as a trigger to unlock the iPad and show the slideshow. If you three finger tap, or pinch, or slide out, switch to the front end by showing the Home Assistant app. The iPad displays a  view which is hidden on my other devices, and only that view, hiding the header using Custom Compact Header. On that view I also have buttons for spotify and netflix, which also sends SSH commands to it, to switch to those app on the iPad, to act as remote controls.\n\nNow, I have used my motion sensors elsewhere, but I have a cheap Xiaofang camera which is hacked and connected to Surveillance Station on my Synology. When it detects motion it sends a http request to a web socket end point in home assistant, on which an automation listens to. When receiving a request, the command line switch is triggered, starting the SSH session on the iPad. A Node RED flow then checks to see if two minutes have passed since last motion was detected, and if so sleep the screen of the iPad.\n\nI will at some point add a smart plug into the equation to make sure the iPad isn't always connected to power for the sake of battery bloat."
print(text)
print(preprocess_text_v2(text))
print('\n\n')

text = "This is a good list. I'd add / update as follows (Bamba is poked, for example):\n\nCity: Aguero, Sterling, Bilva, Ederson (?)\n\nChelsea: Hazard, Higuain, Luiz (?)\n\nBrighton: Duffy\n\nUnited: Pogba, Lukaku (?), Rashford (?), Shaw, DDG (?)\n\nWolves: Doherty, Jimenez, \n\nPalace: Zaha (?), Schlupp (depends on whether they get a double)\n\nWatford: None\n\nSpurs: Son (Kane if they get a double?)\n\nCardiff: None\n\nArsenal: None (they only have a single)\n\nSouthampton: Ings (if fit), Hojbjerg, Bednarek (depends on whether they get a double)\n\n&amp;#x200B;\n\n "
print(text)
print(preprocess_text_v2(text))
print('\n\n')

text = "enter http://facebook.com!!"
print(text)
print(preprocess_text_v2(text))
print('\n\n')





# def prepare_sentence(text, target_arr):

# 	text = text.replace('\n', ' ')
# 	text = text.replace('](https:', '] (https:')
# 	arr = nltk_tokenize_sentence.sent_tokenize(text)
# 	target_arr = [word.lower() for word in target_arr]

# 	for item in arr:
# 		target_join = ' '.join(target_arr)
# 		target_join = ' ' + target_join + ' '
# 		sentence = [tok.text.lower() for tok in nlp.tokenizer(item)]
# 		sentence = ' '.join(sentence)
# 		sentence = ' ' + sentence + ' '