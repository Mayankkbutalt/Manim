class FourierScene(Scene):
    n_vectors = 101
    center_point = ORIGIN
    slow_factor = 0.1

    def get_freqs(self):
        n = self.n_vectors
        all_freqs = list(range(n // 2, -n // 2, -1))
        all_freqs.sort(key=abs)
        return all_freqs

    # Computing Fourier series
    # i.e. where all the math happens
    # taken from Grant Sanderson, 3Blue1Brown
    # https://github.com/3b1b/videos/tree/master/_2019/diffyq/part4
    def get_coefficients_of_path(self, path, n_samples=10000, freqs=None):
        if freqs is None:
            freqs = self.get_freqs()
        dt = 1 / n_samples
        ts = np.arange(0, 1, dt)
        samples = np.array([
            path.point_from_proportion(t)
            for t in ts
        ])
        samples -= self.center_point
        complex_samples = samples[:, 0] + 1j * samples[:, 1]

        result = []
        for freq in freqs:
            riemann_sum = np.array([
                np.exp(-TAU * 1j * freq * t) * cs
                for t, cs in zip(ts, complex_samples)
            ]).sum() * dt
            result.append(riemann_sum)

        return result
    
    def get_path(self):
        tex_mob = MathTex(r"\pi")
        tex_mob.scale_to_fit_height(4)
        path = tex_mob.family_members_with_points()[0]
        path.set_fill(opacity=0)
        path.set_stroke(WHITE, 1)
        return path
        
    def construct(self):
        npl = NumberPlane()
        npl.add_coordinates()
        self.add(npl)

        freqs = self.get_freqs()
        path  = self.get_path()
        coefs = self.get_coefficients_of_path(path, freqs=freqs)

        self.play(Create(path))
        self.wait(2)

        vectors = VGroup()
        origin = ORIGIN
        for i in range(len(freqs)):
            print("{:3.0f}: abs = {:5.3f}  Z = {:-5.3f} + {:-5.3f}j".format(
                freqs[i], 
                np.abs(coefs[i]), 
                np.real(coefs[i]), 
                np.imag(coefs[i])))
            dummy = Line(
                start = ORIGIN, 
                end   = [np.real(coefs[i]), np.imag(coefs[i]), 0]
            ).shift(origin)
            vectors += dummy
            origin = dummy.get_end()
        print(len(vectors))
        self.add(vectors)
        self.wait()
        trace = VMobject().set_points([vectors[-1].get_end()]).set_color(YELLOW)
        self.add(trace)

        def vectorsUpdater(mobj, dt):
            origin = mobj[0].get_end()
            for i in range(1, len(freqs)):
                mobj[i].rotate(2*PI*dt*freqs[i]*self.slow_factor, about_point=mobj[i].get_start()).shift(origin - mobj[i].get_start())
                origin = mobj[i].get_end()
            trace.add_line_to(mobj[-1].get_end())    
        vectors.add_updater(vectorsUpdater)

        self.wait(1/self.slow_factor)

        vectors.remove_updater(vectorsUpdater)
        self.wait(2)



class FourierCirclesScene(Scene):
    n_vectors = 101
    center_point = ORIGIN
    slow_factor = 0.1

    def get_freqs(self):
        n = self.n_vectors
        all_freqs = list(range(n // 2, -n // 2, -1))
        all_freqs.sort(key=abs)
        return all_freqs

    # Computing Fourier series
    # i.e. where all the math happens
    # taken from Grant Sanderson, 3Blue1Brown
    # https://github.com/3b1b/videos/tree/master/_2019/diffyq/part4
    def get_coefficients_of_path(self, path, n_samples=10000, freqs=None):
        if freqs is None:
            freqs = self.get_freqs()
        dt = 1 / n_samples
        ts = np.arange(0, 1, dt)
        samples = np.array([
            path.point_from_proportion(t)
            for t in ts
        ])
        samples -= self.center_point
        complex_samples = samples[:, 0] + 1j * samples[:, 1]

        result = []
        for freq in freqs:
            riemann_sum = np.array([
                np.exp(-TAU * 1j * freq * t) * cs
                for t, cs in zip(ts, complex_samples)
            ]).sum() * dt
            result.append(riemann_sum)

        return result
    
    def get_path(self):
        tex_mob = MathTex(r"\pi")
        tex_mob.scale_to_fit_height(4)
        path = tex_mob.family_members_with_points()[0]
        path.set_fill(opacity=0)
        path.set_stroke(WHITE, 1)
        return path
        
    def construct(self):
        npl = NumberPlane()
        npl.add_coordinates()
        self.add(npl)

        freqs = self.get_freqs()
        path  = self.get_path()
        coefs = self.get_coefficients_of_path(path, freqs=freqs)

        self.play(Create(path))
        self.wait(2)

        vectorsCircles = VGroup()
        origin = ORIGIN
        for i in range(len(freqs)):
            print("{:3.0f}: abs = {:5.3f}  Z = {:-5.3f} + {:-5.3f}j".format(
                freqs[i], 
                np.abs(coefs[i]), 
                np.real(coefs[i]), 
                np.imag(coefs[i])))
            dummy = Line(
                start = ORIGIN, 
                end   = [np.real(coefs[i]), np.imag(coefs[i]), 0]
            )
            circ = Circle(radius=np.abs(coefs[i])).set_stroke(width=1, color=RED)
            vectorsCircles += VGroup(dummy, circ).shift(origin)
            origin = dummy.get_end()

        self.add(vectorsCircles)
        self.wait()
        trace = VMobject().set_points([vectorsCircles[-1][0].get_end()]).set_color(YELLOW)
        self.add(trace)

        def vectorsUpdater(mobj, dt):
            origin = mobj[0][0].get_end()
            for i in range(1, len(freqs)):
                mobj[i][0].rotate(2*PI*dt*freqs[i]*self.slow_factor, about_point=mobj[i][0].get_start())
                mobj[i].shift(origin - mobj[i][0].get_start())
                origin = mobj[i][0].get_end()
            trace.add_line_to(mobj[-1][0].get_end())    
        vectorsCircles.add_updater(vectorsUpdater)

        self.wait(1/self.slow_factor)

        vectorsCircles.remove_updater(vectorsUpdater)
        self.wait(2)



