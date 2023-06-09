#!/usr/bin/env python3
# -*- coding: utf-8 -*-
def run(s1, s2, mML=3, treshold=0.5):
	# compute tiles
	global tiles, matchList
	tiles = []
	matchList = []
	tiles = RKR_GST(s1, s2, mML)
	simResult = calcSimilarity(s1.split(), s2.split(), tiles, treshold)
	return simResult[0]


def RKR_GST(P, T, minimalMatchingLength=3, initsearchSize=20):
	"""Computes Running-Karp-Rabin-Greedy-String-Tiling.

		P pattern string
		T text string

		More information can be found here:
		"String Similarity via Greedy String Tiling and Running
		Karp-Rabin Matching"
		https://www.pam1.bcs.uwa.edu.au/~michaelw/ftp/doc/RKR_GST.ps
		"YAP3: Improved Detection of Similarities in Computer Program
		and other Texts"
		https://www.pam1.bcs.uwa.edu.au/~michaelw/ftp/doc/yap3.ps
	"""
	PList = P.split()
	TList = T.split()
	s = initsearchSize
	# tiles = []
	stop = False
	while not stop:
		# Lmax is size of largest maximal-matches from this scan
		Lmax = scanpattern(s, PList, TList)
		# if very long string no tiles marked. Iterate with larger s
		if Lmax > 2 * s:
			s = Lmax
		else:
			# Create tiles from matches taken from list of queues
			markstrings(s, PList, TList)
			if s > 2 * minimalMatchingLength:
				s = s // 2
			elif s > minimalMatchingLength:
				s = minimalMatchingLength
			else:
				stop = True
	return list(tiles)


def scanpattern(s, P, T):
	"""Scans the pattern and text string lists for matches.

		If a match is found that is twice as big as the search length s
		that size is returned, to be used to restart the scanpattern with it.
		All matches found are stored in a list of matches in queues.
	"""
	longestMaxMatch = 0
	queue = []
	hashtable = GSTHashtable()
	# Starting at the first unmarked token in T
	# for each unmarked Tt do
	# if distance to next tile <= s then
	# advance t to first unmarked token after next tile
	# else create the KR-hash value  for substring Tt to Tt+s-1 and add to hashtable
	t = 0
	noNextTile = False

	while t < len(T):
		if isMarked(T[t]):
			t = t + 1
			continue

		dist = distToNextTile(t, T)
		if dist == None:  # no next Tile was found
			dist = len(T) - t
			noNextTile = True

		if dist < s:  # if dist <= s:
			if noNextTile:
				t = len(T)
			else:
				t = jumpToNextUnmarkedTokenAfterTile(t, T)
				if t == None: t = len(T)  # no next unmarked token after Tile was found
		else:
			substring = "".join(T[t:t + s])  # substring = "".join(T[t:t+dist])
			h = createKRHashValue(substring)
			hashtable.add(h, t)  # save (hashvalue, position)
			t = t + 1

	# Starting at the first unmarked token of P
	# for each unmarked Pp do
	# if distance to next tile <= s then
	# advance p to first unmarked token after next tile
	# else
	# create the KR hash-value for substring Pp to Pp+s-1
	# check hashtable for hash of KR hash-value
	# for each hash-table entry with equal hashed KR hash-value do
	# if for all j from 0 to s-1, Pp+ j = Tt+ j then /* IE match is not hash artifact */
	# k: = s
	# while Pp+k = Tt+k AND unmarked(Pp+k) AND unmarked(Tt+k) do
	# k := k + 1
	# if k > 2 *s then return(k) /* and restart scanpattern with s = k */
	# else record new maximal-match
	noNextTile = False
	p = 0
	while p < len(P):  # for p in range(0, len(P)):
		if isMarked(P[p]):
			p = p + 1
			continue

		dist = distToNextTile(p, P)
		if dist == None:  # no next Tile was found
			dist = len(P) - p
			noNextTile = True

		if dist < s:  # if dist <= s:
			if noNextTile:
				p = len(P)
			else:
				p = jumpToNextUnmarkedTokenAfterTile(p, P)  # TODO:
				if p == None: p = len(P)  # no next unmarked token after Tile was found
		else:
			substring = "".join(P[p:p + s])  # substring = "".join(P[p:p+dist])
			h = createKRHashValue(substring)
			values = hashtable.get(h)
			if values != None:
				for val in values:
					if "".join(T[val:val + s]) == substring:
						t = val
						k = s
						while (p + k < len(P) and t + k < len(T) and
								P[p + k] == T[t + k]
								and isUnmarked(P[p + k]) and isUnmarked(T[t + k])):
							k = k + 1
						if k > 2 * s:
							return k
						else:
							if longestMaxMatch < s: longestMaxMatch = s
							queue.append((p, t, k))
						# recordMaxMatch() #TODO
			p = p + 1
	# add queue to matchList if it is not empty
	if queue != []:
		matchList.append(queue)
	# Return(length of longest maximal-match)
	return longestMaxMatch


def markstrings(s, P, T):
	lengthOfTokenTiled = 0
	# for each queue
	while matchList != []:
		queue = matchList.pop(0)
		while queue != []:  # while queue is not empty
			match = queue.pop(0)  # match = (p-position, t-position, length)
			if not isOccluded(match, tiles):
				for j in range(0, match[2]):
					P[match[0] + j] = markToken(P[match[0] + j])
					T[match[1] + j] = markToken(T[match[1] + j])
				lengthOfTokenTiled = lengthOfTokenTiled + match[2]
				tiles.append(match)


def createKRHashValue(substring):
	"""Creates a Karp-Rabin Hash Value for the given substring
		and returns it.

		Based on:
		https://www-igm.univ-mlv.fr/~lecroq/string/node5.html
	"""
	# ===============================================================================
	#    return hash(substring)
	# ===============================================================================
	hashValue = 0
	for c in substring:  # for i in range(0, len(substring)):
		hashValue = ((hashValue << 1) + ord(c))  # hashValue = ((hashValue<<1) + substring[i])
	return hashValue


def isUnmarked(s):
	"""If string s is unmarked returns True otherwise False.
	"""
	if len(s) > 0:
		return s[0] != '*'
	else:
		return False  # True or False?


def isMarked(s):
	return not isUnmarked(s)


def markToken(s):
	"""Mark string s.
	"""
	return '*' + s


def isOccluded(match, tiles):
	"""Returns true if the match is already occluded by another match
		in the tiles list.

		"Note that "not occluded" is taken to mean that none of the tokens
		Pp to Pp+maxmatch-1 and Tt to Tt+maxmatch-1 has been marked during
		the creation of an earlier tile. However, given that smaller tiles
		cannot be created before larger ones, it suffices that only the ends
		of each new putative tile be testet for occlusion, rather than the whole
		maxmimal match."
		["String Similarity via Greedy String Tiling and Running Karp-Rabin Matching"
		https://www.pam1.bcs.uwa.edu.au/~michaelw/ftp/doc/RKR_GST.ps]
	"""
	for m in tiles:
		if (m[0] + m[2] == match[0] + match[2]
				and m[1] + m[2] == match[1] + match[2]):
			return True
	return False


def distToNextTile(pos, stringList):
	"""Returns distance to next tile, i.e. to next marked token.
		If not tile was found, it returns None.

		case 1: there is a next tile
			-> pos + dist = first marked token
			-> return dist
		case 2: there is no next tile
			-> pos + dist = len(stringList)
			-> return None
		dist is also number of unmarked token 'til next tile
	"""
	# case 2: is last token in list
	if pos == len(stringList): return None

	dist = 0
	while pos + dist + 1 < len(stringList) and isUnmarked(stringList[pos + dist + 1]):
		dist = dist + 1

	# case 2: no tile was found
	if pos + dist + 1 == len(stringList): return None

	# case 1:
	return dist + 1


def jumpToNextUnmarkedTokenAfterTile(pos, stringList):
	"""Returns the first postion of an unmarked token after the next tile.

		case 1: -> normal case
			-> tile exists
			-> there is an unmarked token after the tile
		case 2:
			-> tile exists
			-> but NO unmarked token after the tile
		case 3:
			-> NO tile exists
	"""
	dist = distToNextTile(pos, stringList)
	# case3: No Tile was found
	if dist == None: return None

	pos = pos + dist  # pos on first marked token
	while pos + 1 < len(stringList) and not isUnmarked(stringList[pos + 1]):
		pos = pos + 1

	# case 2: No unmarked Token after Tile was found
	if pos + 1 > len(stringList) - 1: return None

	# case 1:
	return pos + 1


# ===============================================================================
#    Computation of the Similarity
# ===============================================================================
def calcSimilarity(s1List, s2List, tiles, treshold):
	"""Calculates Similarity and returns list [similarity:float, suspectedPlagiarism:bool]"""
	# compute similarity
	similarity = sim(s1List, s2List, tiles)

	# check if it is suspected plagiarism
	suspPlag = similarity >= treshold

	return [similarity, suspPlag]


def sim(A, B, tiles):
	"""Returns similarity value for token of text A and B and the similary tiles covered.
	"""
	return float(2 * coverage(tiles)) / float(len(A) + len(B))


def coverage(tiles):
	"""Sum of length of all tiles.
	"""
	accu = 0
	for tile in tiles:
		accu = accu + tile[2]
	return accu


# ===============================================================================
#    Extra Class: GSTHashtable
# ===============================================================================
class GSTHashtable:

	def __init__(self):
		self.dict = {}

	def add(self, key, ob):
		"""Stores object 'ob' for key 'key' in a list. If there are already
			objects stored in the list for the key -> 'ob' is appended.
		"""
		if key in self.dict:
			values = self.dict.get(key)
			values.append(ob)
			self.dict.setdefault(key, values)
		else:
			self.dict.setdefault(key, [ob])

	def get(self, key):
		"""Returns a list with all objects for key 'key'. If the key does not exist
			'None' is returned.
		"""
		if key in self.dict:
			return self.dict.get(key)
		else:
			return None

	def clear(self):
		"""Clears the GSTHashtable, i.e. all entries are removed.
		"""
		self.dict = {}
