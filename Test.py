import nltk
import ssl
from nltk.corpus import opinion_lexicon
from nltk.stem.porter import PorterStemmer
import math
import numpy
import sklearn
from sklearn.linear_model import LogisticRegression
import pandas
import matplotlib.pyplot as plt

# try:
#     _create_unverified_https_context = ssl._create_unverified_context
# except AttributeError:
#     pass
# else:
#     ssl._create_default_https_context = _create_unverified_https_context

# nltk.download('opinion_lexicon')
# print(289*742)

# str = "What's a good sentence."
# bigram = list(nltk.bigrams(str.split()))
# print(*map(' '.join, bigram), sep=', ')

porter = PorterStemmer()

def lexStemmer(lexicon):
    stemmedLexicon = []
    for word in lexicon:
        stemmedLexicon.append(porter.stem(word))
    return stemmedLexicon

nltkPosLex = opinion_lexicon.positive()
nltkNegLex = opinion_lexicon.negative()
posLex = ["".join(list_of_words) for list_of_words in nltkPosLex]
negLex = ["".join(list_of_words) for list_of_words in nltkNegLex]

stemmedPosLex = set(lexStemmer(posLex))
stemmedNegLex = set(lexStemmer(negLex))
# print(stemmedPosLex)
# print(stemmedNegLex)


# testData = [({'This': False, 'not': False,'good': False, 'sentence':False}, 'pos'), ({'Another': False, 'sentence': False}, 'neg')]
# #
# lrTraining = []


lrTraining = [(['i', 'dont', 'like', 'it'], 'neg'), (['i', 'like', 'it', 'very', 'awesom'], 'pos'), (['terrible', 'product', 'not', 'good'], 'neg')]

def featureCount(review):
    frequencies = [0, 0, 0, 0, 0, 0, 0]
    for word in review[0]:
        if word in stemmedPosLex:
            frequencies[0] += 1
        elif word in stemmedNegLex:
            frequencies[1] += 1
    # restrung = " ".join(review[0])
    # reviewBigrams = list(nltk.bigrams(restrung.split()))
    # for bigram in reviewBigrams:
    #     #print(bigram)
    #     if bigram[0] == 'not' and bigram[1] == 'good':
    #         frequencies[2] += 1
    #     elif bigram[0] == 'i' and bigram[1] == 'like':
    #         frequencies[3] += 1
    #     elif bigram[0] == 'not' and bigram[1] == 'bad':
    #         frequencies[4] += 1
    #     elif bigram[0] == 'dont' and bigram[1] == 'like':
    #         frequencies[5] += 1
    if review[1] == 'pos':
        frequencies[6] = 1
    elif review[1] == 'neg':
        frequencies[6] = 0
    # # print(frequencies)
    print("counted features")
    return frequencies


# #
# # print(lrTraining)
# #
# # td = data(testData)
# # print(td)
#
#
# formattedData = []
# for review in lrTraining:
#     formattedData.append(featureCount(review))
# print(formattedData)
#
# dataFrame = pandas.DataFrame(formattedData, columns=['f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'class'])
# print(dataFrame)
#
# # Get feature columns from dataframe
# X = dataFrame.iloc[:, 0:2]
# y = dataFrame.iloc[:, -1]
#
# print(X)
# print(y)

# logRegression = LogisticRegression(solver='sag')
# logRegression.fit(X, y)
# print("done")

def reformatForLR(nbFormatted):
    reformatted = []
    for rev in nbFormatted:
        bigList = [rev[0].keys(), rev[1]]
        reformatted.append(bigList)
        print(bigList)
    return reformatted

unformatted = [({'outstanding': False, 'hugely': False, 'shoulder': False, 'determining': False, '14000': False, '76s': False, 'lifecams': False, 'accomplished': False, 'fees': False, 'louis': False, 'reenforced': False, 'avaiable': False, 'resync': False, 'sat': False, 'allow': False, 'disgrace': False, 'discovered': False, 'unpresentable': False, '2255': False, 'substituted': False, 'freshertasting': False, 'availab': False, 'drag': False, 'novi': False, 'offices': False, 'box': False, 'reaches': False, 'pry': False, 'outmatch': False, 'microofts': False, 'pedestrians': False, 'db': False, 'diamondback': False, 'faster': False, 'ergonomics': False, 'frying': False, 'outstretched': False, 'thereafter': False, 'seek': False, 'posture': False, 'purple': False, 'skycaddie': False, 'remakes': False, 'readers': False, 'mushes': False, 'pans': False, 'wm5': False, 'retried': False, 'msd': False, '10001': False, 'surface': False, '45min': False, 'helpful': False, 'arrow': False, 'music': False, 'scenes': False, 'session': False, 'timely': False, 'breath': False, 'failure': False, 'advanced': False, 'thumb': False, 'inputs': False, 'keep': False, 'fists': False, 'perfomace': False, '100': False, 'greatly': False, 'this': True, 'odd': False, 'listing': False, 'warmth': False, 'kill': False, 'signaltonoise': False, 'daxon0016s': False, 'waits': False, 'sharpin': False, 'san': False, 'gauge': False, 'lanyard': False, 'partion': False, '500': False, 'plague': False, 'hits': False, 'complains': False, 'smooth': False, 'flammable': False, 'lound': False, 'though': False, 'freely': False, 'outer': False, 'mad': False, 'mone': False, 'temperature': False, 'cords': False, 'commands': False, 'portability': False, 'wear': False, 'supports': False, 'tdk': False, 'backed': False, 'viewpoint': False, '325ci': False, 'binding': False, 'asin': False, 'fixreplaceetc': False, 'previous': False, 'refurbish': False, 'eyeing': False, 'paced': False, 'presenter': False, 'pric': False, 'farthest': False, 'knowmy': False, 'wishes': False, 'bright': False, 'im716': False, 'ge': True, 'faraway': False, 'emp': False, 'laser': False, 'vehiclehighly': False, 'veer': False, 'microsoftcom': False, 'extreamly': False, 'rl': False, '19731974': False, 'undoubtedly': False, 'standard': False, '2499': False, 'sportaproportapro': False, 'harddrives': False, 'camping': False, 'characteristices': False, 'yetthank': False, 'generators': False, 'neverlost': False, 'jsut': False, 'rats': False, 'postage': False, 'thanksgiving': False, 'midtrebley': False, 'taping': False, 'daysafter': False, 'programmable': False, 'sensativity': False, 'wrt54g': False, 'mink': False, 'mutilating': False, 'other': False, 'uncomfortable': False, 'pointwireless': False, 'thumbwheel': False, 'speedy': False, 'sticker': False, 'g7': False, 'criteria': False, 'limit': False, 'phil': False, 'short': False, 'leds': False, 'coincidence': False, '600kbps': False, 'c330': False, 'finalized': False, 'expanded': False, 'hard': False, 'fooling': False, 'executives': False, 'preamp': False, 'gravity': False, 'belief': False, 'bestlooking': False, 'sections': False, 'rendered': False, 'visor': False, 'k240s': False, 'cycling': False, 'partnership': False, 'aiwa': False, 'dive': False, 'paul': False, '20d': False, 'implementation': False, 'novelty': False, 'oyster': False, 'handspring': False, 'junk': False, 'muc': False, 'water': False, 'okit': False, 'sequences': False, 'dumping': False, 'attach': False, 'ugliest': False, 'bash': False, 'certificate': False, 'directly': False, 'prevent': False, 'verify': False, 'tweaked': False, 'acknowledged': False, 'ironed': False, 'desk': False, 'reviews': False, 'compatiblity': False, 'ia': False, 'pads': False, 'busyforward': False, 'primary': False, 'logitech': False, 'lyra': False, 'e30s': False, 'billsso': False, 'firstly': False, 'gfriends': False, 'eyecatching': False, 'monroe': False, 'wap54g': False, 'conditioning': False, 'mere': False, '760': False, 'runing': False, 'comparied': False, 'mux': False, 'opens': False, 'breakdowns': False, 'logged': False, 'reviewsi': False, 'belts': False, 'ship': False, '40x': False, 'fix': False, 'realtime': False, 'wg302': False, 'equally': False, 'quest': False, 'china': False, 'talented': False, 'serveral': False, '350dollar': False, 'send': False, 'children': False, 'overcast': False, 'owning': False, 'aloh': False, 'seamless': False, 'indispensible': False, 'trim': False, '16x': False, 'mice': False, 'boston': False, 'missus': False, 'isits': False, 'h320': False, 'mdrxd100s': False, 'cushiony': False, 'freeze': False, 'nudging': False, '911': False, 'arranged': False, 'changing': False, 'given': False, 'parameters': False, 'logger': False, 'cdromdvdrom': False, 'neck': False, 'tournament': False, 'deliver': False, 'wants': False, 'prevented': False, 'upc': False, 'valley': False, 'dxing': False, 'its': False, 'wep': False, 'visamc': False, 'remained': False, 'indestructable': False, 'fart': False, 'shipping': False, 'inaccessible': False, 'cured': False, 'rpm': False, 'fixedbase': False, 'sexy': False, 'toshiba': False, 'relied': False, 'indicate': False, 'altough': False, 'texture': False, 'five': False, 'matters': False, 'manualoops': False, '2': False, 'rotate': False, 'airplane': False, 'kband': False, 'zooming': False, 'yr': False, 'driving': False, 'piano': False, 'fools': False, 'involve': False, 'theyd': False, 'clauses': False, 'elegant': False, 'dongles': False, 'consumers': False, 'alcohol': False, 'tanking': False, 'wishing': False, '70': False, 'properly': False, 'covers': False, 'images': False, 'vicegripclamp': False, 'quantity': False, 'tilt': False, 'bridging': False, 'timehonored': False, 'market': False, 'crushed': False, 'inexpensive': False, 'configurations': False, 'carrying': False, 'dealbreaker': False, 'altec': False, 'domestic': False, 'esoteric': False, 'retirement': False, 'pictue': False, 'annoys': False, 'runups': False, 'generated': False, 'tma': False, 'wondered': False, 'excel': False, 'sun': False, 'network': False, 'viable': False, 'watching': False, 'uglier': False, 'eleven': False, 'waterproofness': False, 'majority': False, 'charing': False, 'vow': False, 'blink': False, 'hoax': False, 'excessively': False, 'esmartbuy': False, 'fujifilm': False, 'dish': False, 'pa': True, 'unrecoverable': False, 'addiction': False, 'ray': False, 'aim': False, 'posts': False, 'nose': False, 'something': False, 'freeware': False, 'broken': False, 'arch': False, 'budget': False, 'begin': False, 'typist': False, 'hx2755': False, 'dimmed': False, 'sensitivity': False, 'm1000': False, 'lining': False, 'guardianship': False, 'vocals': False, 'kudos': False, 'precise': False, 'blue': False, 'section': False, 'damage': False, 'workouts': False, 'cans': False, 'heaphone': False, 'specify': False, 'rightside': False, 'since': False, 'rumor': False, 'vent': False, 'toasted': False, 'lot': False, 'selfconfigured': False, 'problematic': False, 'otoscope': False, 'glass': False, 'daytoday': False, 'kxtg6700b': False, 'caramel': False, 'sacds': False, '1030': False, 'dying': False, 'sanso': False, 'relavant': False, 'stays': False, 'provider': False, 'joggingexercise': False, '32': False, 'intentional': False, '95pm': False, 'cat5e': False, 'oneway': False, 'nightly': False, 'avalon': False, 'damn': False, 'brezz': False, '5': False, '10inch': False, 'shouldnt': False, 'myrtle': False, 'styed': False, 'afterwards': False, 'printed': False, 'align': False, 'sealed': False, 'done': False, 'smple': False, 'marketing': False, 'practices': False, 'airwaves': False, 'unbelievable': False, 'girlfriend': False, 'hitachi': False, 'calmly': False, 'radio': False, 'lock': False, 'hide': False, 'brought': False, 'vcrdvd': False, 'as': True, 'reasons': False, 'ultraata': False, 'valubles': False, 'steelbar': False, 'motion': False, 'snotty': False, 'firefox': False, 'honey': False, 'sporatic': False, 'c150': False, 'infocus': False, 'bud': False, 'near': False, 'flawlessly': False, '40mm': False, 'muvo': False, 'across': False, 'fr300': False, 'compatability': False, 'boo': False, 'chances': False, 'webinterface': False, 'commonly': False, 'snapped': False, 'karma': False, 'xrocker': False, 'skins': False, 'symmetric': False, 'proof': False, 'oops': False, 'wobble': False, 'zune': False, 'born': False, 'regretting': False, 'snapon': False, 'accidently': False, 'crc': False, 'hijacks': False, 'wife': False, 'interference': False, 'requiered': False, '2018': False, 'downmixed': False, 'sentences': False, 'v3m': False, 'humberto': False, 'foward': False, 'flaking': False, 'appliances': False, 'productwise': False, 'fingers': False, 'afternoon': False, 'drawback': False, 'unstoppable': False, 'hooked': False, 'packaged': False, 'sansa': False, 'soon': False, 'feel': False, 'preventing': False, 'confirmation': False, 'aestethic': False, 'attempting': False, 'doublesize': False, 'routing': False, 'marvelous': False, 'eye': False, 'crossing': False, 'wild': False, 'constant': False, 'sensativities': False, 'rubbery': False, 'microwaves': False, 'interrupted': False, 'ag': False, 'rocky': False, 'restore': False, 'ctrlc': False, 'dull': False, 'respectable': False, 'combined': False, 'annoy': False, 'quartermile': False, 'expences': False, 'connectors': False, 'grown': False, '24gig': False, 'opinio': False, 'frank': False, 'decoder': False, 'portions': False, 'e2c': False, 'gut': False, 'breakdown': False, 'freespace': False, 'occurances': False, 'northwest': False, '4500': False, 'mswhich': False}, 'pos'), ({'exchange': False, 'dependant': False, 'noting': False, 'them': False, 'recover': False, 'mailin': False, 'comparably': False, '5560': False, 'rooms': False, 'tiles': False, 'trickier': False, 'hurry': False, 'meet': False, 'stuffers': False, 'entire': False, 'grundig': False, 'silly': False, 'metropolitan': False, '21quot': False, 'cups': False, 'appreciate': False, 'juice': False, '5075': False, 'styli': False, 'orbits': False, 'netexe': False, 'use': False, 'volumespeakers': False, 'imaging': False, '10cup': False, 'privacy': False, 'recovered': False, 'controller': False, 'xbox360': False, '7050': False, 'starsis': False, 'secured': False, 'x2': False, 'thinner': False, 'boxwavecom': False, 'sbl': False, 'ic': False, 'gummed': False, 'assume': False, 'exchanged': False, 'smells': False, 'glad': False, '6': False, 'veritable': False, 'vcd': False, 'ncaanfl': False, 'occasional': False, 'testament': False, 'reliabilityquality': False, 'grandchildren': False, '149': False, 'interet': False, 'cnd': False, 'astonishing': False, 'system': False, 'powered': False, 'xg': False, 'dusty': False, 'wheels': False, 't5400': False, 'disceted': False, 'megahertz': False, 'screen': False, 'saw': False, 'ii': False, '104': False, 'dramatic': False, 'orbit': False, 'compr\x1a': False, 'twiceinto': False, 'sdusb': False, 'barely': False, 'microphonerecording': False, 'antivirusfirewall': False, '149999': False, 'regularsized': False, 'miscalculated': False, 'communicator': False, 'dvi': False, 'luggageno': False, 'engineered': False, 'macos': False, 'gripe': False, 'sit': False, 'praise': False, 'paws': False, 'cared': False, 'telnet': False, 'clearing': False, 'thickest': False, '883': False, 'wizard': False, 'youre': False, 'update': False, 'disorganized': False, 'straightens': False, 'livable': False, 'eclipse': False, 'obscure': False, 'confident': False, 'denver': False, '845': False, 'flange': False, 'ctrlaltdelete': False, '108m': False, 'criticisms': False, 'noload': False, 'kmart': False, 'overlaptopequipped': False, 'independant': False, 'seams': False, 'divisiones': False, 'suspect': False, 'perceived': False, 'plugnplay': False, 'satelite': False, 'draw': False, 'emailfriendly': False, 'magnetic': False, 'ue': False, 'ip': False, 'audiotv': False, 'favour': False, 'bluetooh': False, 'colour': False, 'warp': False, 'intersection': False, 'invalidate': False, 'chords': False, 'swivel': False, 'dishonest': False, 'unbeatable': False, 'article': False, 'alternate': False, 'leather': False, 'kxtg2700kxtg2720': False, '106': False, 'complain': False, 'resting': False, 'contrary': False, 'corrosion': False, 'multiswitch': False, 'sleeps': False, 'misses': False, 'gold': False, 'highspeed': False, 'veneer': False, 'set': True, 'longest': False, 'writer': False, 'howver': False, '185': False, 'functionssettings': False, 'jerky': False, 'frisbee': False, 'trackman': False, 'rca': False, 'amfm': False, 'evidently': False, 'unlucky': False, 'recharge': False, 'lying': False, 'magical': False, 'tender': False, 'mark': False, 'mt': False, 'calculations': False, 'r30s': False, 'soundgranted': False, 'definently': False, 'almost': False, 'jukebox': False, 'angryi': False, 'magazine': False, 'interacts': False, 'concept': False, 'corner': False, 'delorme': False, 'nanos': False, 'suckered': False, 'shuffle': False, 'shortcut': False, 'workingmuch': False, 'performed': False, 'option': False, 'syn1301b': False, 'subjective': False, '51': False, 'audibly': False, 'graphire': False, 'hubby': False, 'cx7300': False, 'licking': False, 'africa': False, 'voided': False, 'notsogood': False, '09152006': False, 'cheapy': False, 'altogether': False, 'ocz': False, 'vintage': False, 'says': False, 'oscilloscope': False, 'hadnt': False, 'speaks': False, 'regard': False, 'sophisticated': False, 'exercisingit': False, 'developed': False, 'value': False, 'flaviu': False, 'knob': False, 'u': True, 'puncture': False, 'dvdtv': False, 't5720': False, 'lawnmower': False, 'fantastically': False, 'reformatted': False, 'config': False, 'replug': False, 'promises': False, 'everytime': False, 'blaster': False, 'overheat': False, 'outsideso': False, 'work': False, 'imported': False, 'k8vx': False, 'ht27546': False, 'shures': False, '4g': False, 'lazer': False, 'browsing': False, 'scanners': False, 'had': False, 'quibbles': False, 'footprint': False, 'agenda': False, 'sturdy': False, 'proficient': False, 'blatant': False, 'cartoons': False, 'educational': False, 'contemplating': False, 'manner': False, 'tiresome': False, 'context': False, 'later': False, 'thud': False, 'mot': False, 'perhaps': False, '80mm': False, 'persistance': False, 'beautifullworks': False, 'leaf': False, 'braindead': False, 'extender': False, 'insensitive': False, 'responding': False, 'stepwise': False, 'electical': False, 'mask': False, 'ny': False, 'unpowered': False, 'interfacemenu': False, 'count': False, 'spacialstereo': False, 'technicians': False, 'fabulous': False, 'different': False, 'paperweight': False, 'hints': False, 'morter': False, 'shell': False, 'againthanks': False, 'marginal': False, 'keyboardmouse': False, 'refute': False, 'bt': False, 'relies': False, 'log': False, 'lifetime': False, 'cord': False, 'seal': False, 'incident': False, 'transferred': False, 'grad': False, 'simplicity': False, 'costco': False, 'sync': False, 'nulooq': False, 'conslike': False, '11': False, 'degrades': False, 'smoke': False, 'whe': False, 'constrict': False, 'stationed': False, 'frederic': False, 'gadget': False, 'veiw': False, 'mx5000': False, 'reads': False, 'abroad': False, 'cults': False, 'vacuum': False, 'wisconsin': False, 'kittens': False, 'honor': False, 'hoover': False, 'reducing': False, 'instances': False, 'overbrassy': False, 'ridges': False, 'doityourself': False, 'track': False, 'skipdring': False, 'pristine': False, 'decreased': False, 'fatal': False, 'thats': False, 'posting': False, 'ridata': False, 'nonexistant': False, 'dearly': False, 'luckily': False, '60c': False, 'enoughs': False, '1973': False, 'match': False, 'straight': False, 'home': False, 'enroute': False, 'content': False, 'margin': False, 'huh': False, 'integrated': False, 'remodeling': False, 'addon': False, 'settle': False, 'voltage': False, 'sytem': False, 'push': False, '119': False, 'baffles': False, 'disables': False, 'wirelessg': False, 'chintzy': False, 'enjoys': False, 'dimly': False, 'avi': False, 'intialize': False, 'coordinates': False, 'agents': False, 'imitates': False, 'successfully': False, 'trivially': False, 'randomly': False, 'k7': False, 'waterproof': False, 'ranks': False, 'xfi': False, 'cx2610': False, 'present': False, '37inch': False, 'surprising': False, 'logging': False, 'lamps': False, 'aesthetics': False, 'flexible': False, 'toledo': False, 'promising': False, '60gb': False, 'rs': False, 'mousepads': False, 'jeans': False, '14995': False, 'invisible': False, '80211': False, 'fighting': False, 'proboem': False, 'dialogue': False, 'carafe': False, 'f3545': False, 'mechanics': False, 'understandable': False, 'replace': False, 'shriek': False, 'irritant': False, 'admitted': False, 'pessimistic': False, 'allowing': False, 'guarentee': False, 'recess': False, 'rehoboth': False, 'boiling': False, 'cite': False, 'offending': False, 'dwlg650': False, 'campion': False, 'connect': False, 'tinnier': False, 'bordered': False, 'linksys': False, 'arrogant': False, 'mabey': False, 'cleaner': False, 'bar': False, 'gpsmap': False, 'shielding': False, 'old': False, '600': False, 'cablesthe': False, '1635mm': False, 'texttospeech': False, 'inlets': False, 'oversight': False, 'couple': False, 'pause': False, 'kingston': False, 'tendonitis': False, 'sencillo': False, '48': False, 'usless': False, 'winter': False, 'weaks': False, '1982': False, '1295': False, 'knockoffs': False, 'before': False, 'sensible': False, 'win2000': False, 'visa': False, 'nominal': False, 'h': True, '155s': False, 'steel': False, 'advertised': False, 'lexars': False, 'gibberish': False, 'decent': False, 'cradled': False, 'fills': False, 'fixing': False, 'asymmetric': False, 'doubleseam': False, 'hurts': False, 'rapid': False, 'suit': False, '7lbs': False, 'buttonw': False, 'evening': False, 'alternatives': False, 'mat': True, 'heavyduty': False, 'trusting': False, 'hyped': False, 'desktops': False, 'dismal': False, 'giving': False, 'operated': False, 'w2k': False, 'century': False, 'uninstaller': False, 'enable': False, 'ipodmp3': False, 'cross': False, 'emergency': False, 'kira': False, 'assign': False, 'iskin': False, 'disgusted': False, 'slower': False, 'accomodate': False, 'griffin': False, 'increased': False, 'inactive': False, 'homeand': False, '27quot': False, 'turnaround': False, 'splotchy': False, 'wlitx4g54hp': False, 'thickness': False, 'payback': False, 'searches': False, 'woefully': False, 'pinch': False, 'newly': False, 'beautiful': False, 'paddedi': False, 'target': False, 'recalibration': False, 'seattle': False, 'orgaized': False, 'goodness': False, 'medialife': False, 'estimates': False, 'heavy': False, 'higher': False, 'submit': False, 'vac': False, '6010': False, 'ti': True, 'highpitched': False, 'hollow': False, 'clubs': False, 'jump': False, 'positioned': False, 'sustain': False, 'geforce': False, 'strange': False, 'hissing': False, 'misplaced': False, '1989': False, '34000': False, 'knock': False, '2pieces': False, 'tunnel': False, 'cams': False, 'winwin': False, 'photographs': False, 'hoe': False, 'freq': False, 'tekxon': False, 'staples': False, 'behind': False, 'surely': False, 'csrssexe': False, 'hole': False, '913ns': False, 'interactive': False, 'prism': False, 'passing': False, 'thoughtfully': False, 'dashmount': False, 'suggested': False, 'fell': False, 'fabulousso': False, 'undestand': False, 'mobilebase': False, 'although': False, 'index': False, 'velocity': False, '98': False, 'fallen': False, 'attachedthe': False, 'storm': False, 'essence': False, 'waves': False, 'navigational': False, 'suitcase': False, 'screwdriver': False, 'press': False, 'e3c': False, 'moreon': False, 'eek': False, 'earpads': False, 'arriving': False, 'about': False, 'attractive': False, 'forbid': False, 'overemphasized': False, 'bogus': False, 'bench': False, 'breakage': False, 'executed': False, 'specifically': False, 'luck': False, 'tour': False}, 'neg')]
reformatForLR(unformatted)
formattedData = []
for review in unformatted:
    formattedData.append(featureCount(review))
print(formattedData)

dataFrame = pandas.DataFrame(formattedData, columns=['f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'class'])
print(dataFrame)

