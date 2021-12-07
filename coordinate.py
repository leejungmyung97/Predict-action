import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


class XYZV:
	def __init__(self, x: float, y: float, z: float, v: float) -> None:
		self.update(x, y, z, v)

	def __str__(self) -> str:
		return 'x: {}, y: {}, z: {}, v: {}'.format(round(self.x, 3), round(self.y, 3), round(self.z, 3), round(self.v, 3))

	def __sub__(self, other) -> tuple:
		vx = self.x - other.x
		vy = self.y - other.y
		vz = self.z - other.z
		return (round(vx, 3), round(vy, 3), round(vz, 3))

	def update(self, x, y, z, v):
		self.x = x
		self.y = y
		self.z = z
		self.v = v


class LandMarkList:
	def __init__(self, landmark) -> None:
		self.is_list_1 = True
		self.list_1 = []
		self.list_2 = []
		for i in range(0, 33):
			self.list_1.append(XYZV(0, 0, 0, 0))
			self.list_2.append(XYZV(0, 0, 0, 0))
		self.updateList(landmark)

	def getXYZV(self, idx: int) -> XYZV:
		if self.is_list_1: return self.list_2[idx]
		else: return self.list_1[idx]

	def updateList(self, landmark) -> None:
		if landmark == None: return None
		poses = str(landmark).split('landmark')
		for idx in range(0, 33):
			pose = poses[idx+1].split('\n')
			x = float(pose[1].strip()[3:])
			y = float(pose[2].strip()[3:])
			z = float(pose[3].strip()[3:])
			v = float(pose[4].strip()[13:])
			if self.is_list_1: self.list_1[idx].update(x, y, z, v)
			else: self.list_2[idx].update(x, y, z, v)
			self.is_list_1 = not self.is_list_1

	def getMoveVec(self, idx) -> tuple:
		if self.list_1[idx].v < 0.5 or self.list_2[idx].v < 0.5: return (0, 0, 0)

		if not self.is_list_1: return self.list_2[idx] - self.list_1[idx]
		else: return self.list_1[idx] - self.list_2[idx]

	def distance2points(self, idx1, idx2) -> float:
		if self.is_list_1:
			po1 = self.list_2[idx1]
			po2 = self.list_2[idx2]
		else:
			po1 = self.list_1[idx1]
			po2 = self.list_1[idx2]
		vx, vy, vz = po1-po2
		return (vx**2 + vy**2 + vz**2)**0.5

	def getLandMarkList(self):
		if self.is_list_1: return self.list_2
		else: return self.list_1


class Commander:
	def __init__(self, landMarkList: LandMarkList) -> None:
		self.LML = landMarkList

	def getCommand(self) -> str:
		sholder_scalr = 50.0/self.LML.distance2points(12,13)
		nose = self.LML.getXYZV(0)
		left_hand = self.LML.getXYZV(15)
		right_hand = self.LML.getXYZV(16)
		
		
		if min([nose.v, left_hand.v, right_hand.v]) < 0.7: return 'NoCommand'
		if max([nose.y, left_hand.y, right_hand.y]) == nose.y: return 'Land'
		elif nose.y > left_hand.y:
			vec = self.LML.getMoveVec(16)
			max_v = 5/sholder_scalr
			max_idx = -1
			for i in range(0,2):
				if max_v < abs(vec[i]):
					max_v = abs(vec[i])
					max_idx = i

			if max_idx == 0: # move x axis 
				if vec[max_idx] < 0: return 'cw'
				else: return 'ccw' 
			elif max_idx == 1: # move z axis 
				if vec[max_idx] < 0: return 'forward'
				else: return 'back'

		elif nose.y > right_hand.y:
			vec = self.LML.getMoveVec(15)
			max_v = 5/sholder_scalr
			max_idx = -1
			for i in range(0,2):
				if max_v < abs(vec[i]):
					max_v = abs(vec[i])
					max_idx = i

			if max_idx == 0: # move x axis 
				if vec[max_idx] < 0: return 'right'
				else: return 'left' 
			elif max_idx == 1: # move y axis 
				if vec[max_idx] < 0: return 'up'
				else: return 'down'
			# elif max_idx == 2: # move z axis 
			# 	if vec[max_idx] < 0: return 'back'
			# 	else: return 'forward'
		return 'fail'

class Pose:
	def __init__(self, landMakrList: LandMarkList, model) -> None:
		self.lml_object = landMakrList 
		self.pose = np.zeros(32, dtype=np.float64)
		self.model = model
		self.land2poseMap = [28, 26, 24, 23, 25,
                       27, -1, -2, -3, -4, 16, 14, 12, 11, 13, 15]
		self.land2poseMap2 = [28, 26, 24, 23, 25, 27, 16, 
							14, 12, 11, 13, 15, 0]

	def updateMark(self, idx: int, coord: XYZV):
		
		self.pose[idx] = coord.x
		self.pose[idx+1] = coord.y
	
	def updatePose(self):
		"""		
			0	r ankle X  28
			1	r ankle_Y	
			2	r knee_X	26
			3	r knee_Y	
			4	r hip_X	24
			5	r hip_Y	
			6	l hip_X	23
			7	l hip_Y	
			8	l knee_X	25
			9	l knee_Y	
			10	l ankle_X	 27
			11	l ankle_Y	
			12	pelvis_X	골반 -1
			13	pelvis_Y	
			14	thorax_X	명치 -2
			15	thorax_Y	
			16	upper neck_X	-3
			17	upper neck_Y	
			18	head top_X	-4
			19	head top_Y	
			20	r wrist_X	16
			21	r wrist_Y	
			22	r elbow_X	14
			23	r elbow_Y	
			24	r shoulder_X	12
			25	r shoulder_Y	
			26	l shoulder_X	11
			27	l shoulder_Y	
			28	l elbow_X	 13
			29	l elbow_Y
			30	l wrist_X	15
			31	l wrist_Y
		"""
		lml = self.lml_object.getLandMarkList()

		for x, idx in enumerate(self.land2poseMap):
			# r ankle_X	28

			if idx < 0:
				if idx == -1: # pelvis
					x = (lml[23].x + lml[24].x)/2
					y = (lml[23].y + lml[24].y)/2
					self.pose[12] = x
					self.pose[13] = y
					
				elif idx == -2: # thorax
					x = sum([self.pose[12],lml[11].x, lml[12].x])/3
					y = sum([self.pose[12],lml[11].y, lml[12].y])/3
					self.pose[14] = x
					self.pose[15] = y
					
				elif idx == -3: # upper neck
					lip_x = (lml[10].x + lml[9].x)/2
					lip_y = (lml[10].y + lml[9].y)/2
					x = -(lml[0].x-2*lip_x)
					y = -(lml[0].y-2*lip_y)
					self.pose[16] = x
					self.pose[17] = y
					
				elif idx == -4: # head top
					x = -(self.pose[16]-2*lml[0].x)
					y = -(self.pose[17]-2*lml[0].y)
					self.pose[18] = x
					self.pose[19] = y
			else:
				self.updateMark(x*2, lml[idx])

	def updatePose2(self):
		lml = self.lml_object.getLandMarkList()

		for x, idx in enumerate(self.land2poseMap2):
			self.updateMark(x*2, lml[idx])

	def getPose(self) -> str:
		pose_df = pd.DataFrame([self.pose])
		
		scaled =  pd.DataFrame(StandardScaler().fit_transform(pose_df.transpose())).transpose()
		
		return self.model.predict(scaled)[0]

	 

