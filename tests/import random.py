import random

mylist = [0, 1]
resultat = random.choices(mylist, weights=[70, 30], k=1)[0]
print(resultat)


""" 
        while self.diff_counter < self.difficulty :
            #Vérifions si dans deux cases nous arrivons dans le bord de la fenêtre

            if self.elements[-1][2] == 0 : #La dernière route posée est une ligne droite 

                #if self.actual_coord_ij[0] == np.shape(self.coord)[0] - 2  :#or self.actual_coord_ij[0] == 2 : #On est proche du bord (resp. droite/gauche)
                if self.elements[-1][1] == 90 or  self.elements[-1][1] == 270: #La dernière route posée est une ligne droite dans la direction droite-gauche

                    self.actual_coord_ij = (rd(self.elements[-1][3][0] + 1, self.elements[-1][3][0] - 1), self.elements[-1][3][1]) #La pprochaine case a la même coordonnée y
                    self.actual_pos = self.coord[self.actual_coord_ij[0], self.actual_coord_ij[1]]                   
                    if self.actual_coord_ij[0] == np.shape(self.coord)[0] - 2  :#or self.actual_coord_ij[0] == 2 : #On est proche du bord droit
                        if self.actual_coord_ij[0] == self.elements[-1][3][0] + 1 :
                            self.actual_road_type = 1 #Si on progresse vers le bord, alors on va mettre un virage pour la cohérence du circuit
                            self.actual_angle = rd(90,180) #completion par la droite d'une ligne droite 
                            self.diff_counter += self.diff_curved
                        else :
                            self.actual_road_type = rd(0,1)
                            if self.actual_road_type == 0 :
                                self.actual_angle = rd(90,270)
                                self.diff_counter += self.diff_straight
                                
                            else :
                                self.atual_angle = rd(0,270)      #completion par la gauche d'une ligne droite     
                                self.diff_counter += self.diff_curved

                    elif self.actual_coord_ij[0] == np.shape(self.coord)[0] - 2  :
                        if self.actual_coord_ij[0] == self.elements[-1][3][0] - 1 :
                            self.actual_road_type = 1 #Si on progresse vers le bord, alors on va mettre un virage pour la cohérence du circuit
                            self.actual_angle = rd(0,270) #completion par la gauche d'une ligne droite 
                            
                        else :
                            self.actual_road_type = rd(0,1)
                            if self.actual_road_type == 0 :
                                self.actual_angle = rd(90,270)
                                self.diff_counter += self.diff_straight
                                
                            else :
                                self.atual_angle = rd(90,180)      #completion par la gauche d'une ligne droite     
                                self.diff_counter += self.diff_curved
                    else :
                        self.actual_road_type = rd(0,1)
                        if self.actual_road_type == 0 :
                            self.actual_angle = rd(90,270)
                            self.diff_counter += self.diff_straight
                        else :
                            if self.actual_coord_ij[0] == self.elements[-1][3][0] - 1 :
                                self.actual_angle = rd(0,270)
                            else :
                                self.actual_angle = rd(90, 180)
                            self.diff_counter += self.diff_curved
                else :  #La dernière route posée est une ligne droite dans la direction haut-bas

                    self.actual_coord_ij = (self.elements[-1][3][0], rd(self.elements[-1][3][1] + 1, self.elements[-1][3][1] - 1),) #La pprochaine case a la même coordonnée x
                    self.actual_pos = self.coord[self.actual_coord_ij[0], self.actual_coord_ij[1]]                   
                    if self.actual_coord_ij[1] == 2   :#or self.actual_coord_ij[0] == 2 : #On est proche du bord haut
                        if self.actual_coord_ij[1] == self.elements[-1][3][1] + 1 :
                            self.actual_road_type = 1 #Si on progresse vers le bord, alors on va mettre un virage pour la cohérence du circuit
                            self.actual_angle = rd(90,180) #completion par la droite d'une ligne droite 
                            self.diff_counter += self.diff_curved
                        else :
                            self.actual_road_type = rd(0,1)
                            if self.actual_road_type == 0 :
                                self.actual_angle = rd(90,270)
                                self.diff_counter += self.diff_straight
                                
                            else :
                                self.atual_angle = rd(0,270)      #completion par la gauche d'une ligne droite     
                                self.diff_counter += self.diff_curved

                    elif self.actual_coord_ij[0] == np.shape(self.coord)[0] - 2  :
                        if self.actual_coord_ij[0] == self.elements[-1][3][0] - 1 :
                            self.actual_road_type = 1 #Si on progresse vers le bord, alors on va mettre un virage pour la cohérence du circuit
                            self.actual_angle = rd(0,270) #completion par la gauche d'une ligne droite 
                            
                        else :
                            self.actual_road_type = rd(0,1)
                            if self.actual_road_type == 0 :
                                self.actual_angle = rd(90,270)
                                self.diff_counter += self.diff_straight
                                
                            else :
                                self.atual_angle = rd(90,180)      #completion par la gauche d'une ligne droite     
                                self.diff_counter += self.diff_curved
                    else :
                        self.actual_road_type = rd(0,1)
                        if self.actual_road_type == 0 :
                            self.actual_angle = rd(90,270)
                            self.diff_counter += self.diff_straight
                        else :
                            if self.actual_coord_ij[0] == self.elements[-1][3][0] - 1 :
                                self.actual_angle = rd(0,270)
                            else :
                                self.actual_angle = rd(90, 180)
                            self.diff_counter += self.diff_curved
                    
                



                if self.actual_coord_ij[0] <= 2 : #On est proche du bord gauche
                    if self.actual_coord_ij[1] >= np.shape(self.coord)[1] - 2 : #On est dans un coin
                
                 """