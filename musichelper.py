from six.moves import cPickle as pickle
import numpy as np
from sets import Set 
from collections import OrderedDict
with open('members.csv', 'r') as fin:
	users = []
	user_to_index = {}
	user_song_dic = dict()
	for user_index,line in enumerate(fin):
		if user_index != 0:
			thisLine = line.split(",")
			users.append(thisLine[0])
			user_to_index[thisLine[0]] = user_index - 1
			user_song_dic[user_index-1] = []
	fin.close()
	# print("NUM_USER" + str(len(users)))

with open('songs.csv', 'r') as fin:
	songs = []
	song_to_index = {}
	song_to_genre = {}
	song_to_artist = {}
	song_to_composer = {}
	song_to_language = {}
	song_to_lyricist = {}
	for song_index,line in enumerate(fin):
		if (song_index) != 0:
			thisLine = line.split(",")
			songs.append(thisLine[0])
			song_to_index[thisLine[0]] = song_index - 1
			song_to_genre[thisLine[0]] = thisLine[2]
			song_to_artist[thisLine[0]] = thisLine[3]
			song_to_composer[thisLine[0]] = thisLine[4]
			song_to_language[thisLine[0]] = thisLine[6]
			song_to_lyricist[thisLine[0]] = thisLine[5]

	fin.close()
	# print("NUM_ITEM" + str(len(songs)))

#not here because need to redistribute with users with greater than 2 songs. if u did it here can't know size for the number of songs each user has
with open('train.csv', 'r') as fin:
	for idx,line in enumerate(fin):
		if idx != 0:
			thisLine = line.split(",")
			if thisLine[0] in user_to_index and thisLine[1] in song_to_index and thisLine[5]:	
				user_index = user_to_index[thisLine[0]]
				user_song_dic[user_index].append(thisLine)

	fin.close()


# with open('music/test.csv', 'r') as fin:
# 	pos_user_song_test_dic = {}
# 	for idx,line in enumerate(fin):
# 		thisLine = line.split(",")
# 		if idx != 0:
# 			if thisLine[1] in user_to_index and thisLine[2] in song_to_index:
# 				user_index = user_to_index[thisLine[1]]
# 				song_index = song_to_index[thisLine[2]]
# 				pos_user_song_test_dic[user_index] = Set([song_index])
# 	fin.close() 7377416




next_user_id = 0
train_structured_arr = np.zeros(29823, dtype=[('song_id',np.object, 100), 
                                                ('source_type', np.object, 100),
												('source_system_tab', np.object, 100),
                                                ('source_screen_name', np.object, 100),
                                                ('artist', np.object, 100),
                                                ('genre', np.object, 100),
                                                ('language', np.object, 100),
                                                ('lyricist', np.object, 100),
                                                ('composer', np.object, 100),
                                                ('user_id', np.int32),
                                                ('imp_song_id',np.object), 
												('imp_artist', np.object),
                                                ('imp_genre', np.object),
                                                ('imp_language', np.object),
                                                ('imp_lyricist', np.object),
                                                ('imp_composer', np.object),
                                                ('imp_source_type', np.object),
                                                ('imp_labels',np.object)]) 
val_structured_arr = np.zeros(29164, dtype=[('song_id',np.object, 100), 
												('artist', np.object, 100),
                                                ('genre', np.object, 100),
                                                ('language', np.object, 100),
                                                ('lyricist', np.object, 100),
                                                ('composer', np.object, 100),
                                                ('source_type', np.object, 100),
												('source_system_tab', np.object, 100),
                                                ('source_screen_name', np.object, 100),
                                                ('user_id', np.int32),
                                                ('imp_song_id',np. object), 
												('imp_artist', np.object),
                                                ('imp_genre', np.object),
                                                ('imp_language', np.object),
                                                ('imp_lyricist', np.object),
                                                ('imp_composer', np.object),
                                                ('imp_source_type', np.object),
                                                ('imp_labels',np.object)])
test_structured_arr = np.zeros(29164, dtype=[('song_id',np.object, 100), 
												('artist', np.object, 100),
                                                ('genre', np.object, 100),
                                                ('language', np.object, 100),
                                                ('lyricist', np.object, 100),
                                                ('composer', np.object, 100),
												('source_system_tab', np.object, 100),
                                                ('source_screen_name', np.object, 100),
                                                ('source_type', np.object, 100),
                                                ('user_id', np.int32),
                                                ('imp_song_id',np.object), 
												('imp_artist', np.object),
                                                ('imp_genre', np.object),
                                                ('imp_language', np.object),
                                                ('imp_lyricist', np.object),
                                                ('imp_composer', np.object),
                                                ('imp_source_type', np.object),
                                                ('imp_labels',np.object)])

	

# val_structured_arr = np.zeros(29164, dtype=[('user_history', np.int32, 10), ('item_id', np.int32)]) 
# test_structured_arr = np.zeros(29164, dtype=[('user_history', np.int32, 10), ('item_id', np.int32)])
# users_with_more_than_2_songs =0
# interaction_ind = 0
# for user, songs in user_song_dic.iteritems():
# 	for ind, song_idx in enumerate(list(songs)):
# 		if ind == 0 and len(songs) > 2: 
# 		    val_structured_arr[users_with_more_than_2_songs] = (user, song_idx)
# 		    users_with_more_than_2_songs += 1
# 		elif ind == 1 and len(songs) > 2:
# 		    test_structured_arr[users_with_more_than_2_songs] = (user, song_idx)
# 		else:
# 		    train_structured_arr[interaction_ind] = (user, song_idx)
# 		    interaction_ind += 1




def getInfo(lstSongs, featureType, hist):
	if hist:
		user_Hst = np.zeros(100, dtype=np.object)	 
		if featureType == 'song_id':
			i = 0
			while(i < len(lstSongs)):
				user_Hst[i] = lstSongs[i][1]
				i+= 1

		elif featureType == 'source_type':
			i = 0
			while(i < len(lstSongs)):
				user_Hst[i] = lstSongs[i][4]
				i+= 1

		elif featureType == 'source_system_tab':
			i = 0
			while(i < len(lstSongs)):
				user_Hst[i] = lstSongs[i][2]
				i+= 1

		elif featureType == 'source_screen_name':
			i = 0
			while(i < len(lstSongs)):
				user_Hst[i] = lstSongs[i][3]
				i+= 1

		elif featureType == 'artist':

			i = 0
			while(i < len(lstSongs)):
				# print('==========================================')
				# print(lstSongs[i][1])
				# print(lstSongs[i][1] in song_to_artist)
				# print('==========================================')
				user_Hst[i] = song_to_artist[lstSongs[i][1]] if lstSongs[i][1] in song_to_artist else ''
				i+=1

		elif featureType == 'genre':
			i = 0
			while(i < len(lstSongs)):
				user_Hst[i] = song_to_genre[lstSongs[i][1]] if lstSongs[i][1] in song_to_artist else ''
				i+=1

		elif featureType == 'language':
			i = 0
			while(i < len(lstSongs)):
				user_Hst[i] = song_to_language[lstSongs[i][1]] if lstSongs[i][1] in song_to_artist else ''
				i+=1


		elif featureType == 'lyricist':
			i = 0
			while(i < len(lstSongs)):
				user_Hst[i] = song_to_lyricist[lstSongs[i][1]] if lstSongs[i][1] in song_to_artist else ''
				i+=1

		elif featureType == 'composer':
			i = 0
			
			while(i < len(lstSongs)):
				user_Hst[i] = song_to_composer[lstSongs[i][1]] if lstSongs[i][1] in song_to_artist else ''
				i+=1
				
		return user_Hst
	
	else: 
		userImp = np.empty(1, dtype=object)
		if featureType == 'imp_song_id':
			userImp[0] = lstSongs[1]

		elif featureType == 'imp_artist':
			userImp[0] = song_to_artist[lstSongs[1]] if lstSongs[1] in song_to_artist else ''
		
		elif featureType == 'imp_genre':
			userImp[0] = song_to_genre[lstSongs[1]] if lstSongs[1] in song_to_artist else ''

		elif featureType == 'imp_language':
			userImp[0] = song_to_language[lstSongs[1]] if lstSongs[1] in song_to_artist else ''			

		elif featureType == 'imp_lyricist':

			userImp[0] = song_to_lyricist[lstSongs[1]] if lstSongs[1] in song_to_artist else ''			

		elif featureType == 'imp_composer':
			# print('==========================================')
			# print(lstSongs[1])
			# print(lstSongs[1] in song_to_composer)
			# print('==========================================')
			userImp[0] = song_to_composer[lstSongs[1]] if lstSongs[1] in song_to_artist else ''			

		elif featureType == 'imp_labels':
			userImp[0] = lstSongs[5]

		return userImp


# print("A")

val_idx = 0
test_idx = 0
interaction_ind = 0

moreThan3 = 0
just1 = 0
total = 0
user_id = 0
for user, songs in user_song_dic.iteritems():
	total += 1
	user_history = []
	if len(songs) > 3 and len(songs) < 101:
		moreThan3+=1
		#print('userhistory')
		#print(len(user_history))
		while len(songs) > 3:
			user_history.append(songs.pop())
		# print('userhistory after')
		# print(len(user_history))
		#generate user history for this user
		song_id = getInfo(user_history, 'song_id', 1) 
		source_type = getInfo(user_history, 'source_type', 1)
		source_system_tab = getInfo(user_history, 'source_system_tab', 1)
		source_screen_name = getInfo(user_history, 'source_screen_name', 1)
		artist = getInfo(user_history, 'artist', 1)
		genre = getInfo(user_history, 'genre', 1)
		language = getInfo(user_history, 'language', 1)
		lyricist = getInfo(user_history, 'lyricist', 1)
		composer = getInfo(user_history, 'composer', 1)

		# print(song_id)
		# print(source_type)
		# print(source_system_tab)
		# print(source_screen_name)
		# print(artist)
		# print(genre)
		# print(language)
		# print(lyricist)
		# print(composer)
		for ind in range(len(songs)):
			#For each of the three songs that are not in the user history add them as an impression
			#print('progress5')
			# print(songs)
			# print(songs[ind])

			imp_song_id = getInfo(songs[ind], 'imp_song_id', 0)
			imp_artist = getInfo(songs[ind], 'imp_artist', 0)
			imp_genre = getInfo(songs[ind], 'imp_genre',0)
			imp_language = getInfo(songs[ind], 'imp_language', 0)
			imp_lyricist = getInfo(songs[ind], 'imp_lyricist', 0)
			imp_composer = getInfo(songs[ind], 'imp_composer', 0 )
			imp_source_type = getInfo(songs[ind], 'imp_source_type', 0)
			imp_labels = getInfo(songs[ind], 'imp_labels', 0)
			#print('progress6')
			# print(imp_song_id)
			# print(imp_artist)
			# print(imp_genre)
			# print(imp_language)
			# print(imp_lyricist)
			# print(imp_composer)
			# print(imp_source_type)
			# print(imp_labels)
			if ind == 0: 
			    val_structured_arr[val_idx] = (song_id, source_type, source_system_tab, source_screen_name, 
			    													artist, genre, language, lyricist,composer, user_id, 
			    													imp_song_id, 
			    													imp_artist, 
			    													imp_genre, 
			    													imp_language, 
			    													imp_lyricist, 
			    													imp_composer, 
			    													imp_source_type,
			    													imp_labels
			    													)
			    val_idx += 1
			elif ind == 1:
			    test_structured_arr[test_idx] = (song_id, source_type, source_system_tab, source_screen_name, 
			    													artist, genre, language, lyricist,composer, user_id,
			    													imp_song_id, imp_artist, 
			    													imp_genre, 
			    													imp_language, 
			    													imp_lyricist, 
			    													imp_composer, 
			    													imp_source_type, 
			    													imp_labels
			    													)
			    test_idx += 1 
			else:
			    train_structured_arr[test_idx] = (song_id, source_type, source_system_tab, source_screen_name, 
			    													artist, genre, language, lyricist,composer, user_id,
			    													imp_song_id, 
			    													imp_artist, 
			    													imp_genre, 
			    													imp_language, 
			    													imp_lyricist, 
			    													imp_composer, 
			    													imp_source_type, 
			    													imp_labels)
			    interaction_ind += 1
	elif len(songs) > 1 and len(songs) < 101:
		#2 songs are the history and one song is the training point 
		#print('progress7')
		just1+=1 
		song_id = getInfo(songs[1:], 'song_id', 1) 
		source_type = getInfo(songs[1:], 'source_type', 1)
		imp_labels = getInfo(songs[1:], 'labels', 1)
		source_system_tab = getInfo(songs[1:], 'source_system_tab', 1)
		source_screen_name = getInfo(songs[1:], 'source_screen_name', 1)
		artist = getInfo(songs[1:], 'artist', 1)
		genre = getInfo(songs[1:], 'genre', 1)
		language = getInfo(songs[1:], 'language', 1)
		lyricist = getInfo(songs[1:], 'lyricist', 1)
		composer = getInfo(songs[1:], 'composer', 1 )

		# print('progress8')
		imp_song_id = getInfo(songs[0], 'imp_song_id', 0)
		imp_artist = getInfo(songs[0], 'imp_artist', 0)
		imp_genre = getInfo(songs[0], 'imp_genre', 0)
		imp_language = getInfo(songs[0], 'imp_language', 0)
		imp_lyricist = getInfo(songs[0], 'imp_lyricist', 0)
		imp_composer = getInfo(songs[0], 'imp_composer', 0)
		imp_source_type = getInfo(songs[0], 'imp_source_type', 0)
		# print(imp_labels)
		train_structured_arr[interaction_ind] = (song_id, source_type, source_system_tab, 
														    source_screen_name, artist, genre, language, 
														    lyricist,composer, user_id, imp_song_id, imp_artist, 
														    imp_genre, imp_language, imp_lyricist, 
														    imp_composer, imp_source_type, imp_labels)
		interaction_ind += 1
	user_id += 1
	if total % 1000 == 0: print(total)
	if user_id == 11: break 



print('saving train struct')
np.save('train_structured_arr_batch', train_structured_arr)
print('saving val struct_batch')
np.save('val_structured_arr_batch', val_structured_arr)
print('saving test struct')
np.save('test_structured_arr_batch', test_structured_arr) 


print('========================================================')