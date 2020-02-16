import twint

c = twint.Config()
c.Store_csv = True
c.Output = ('./trump.csv')
c.Limit = 1000
c.Username = 'realdonaldtrump'
twint.run.Search(c)
