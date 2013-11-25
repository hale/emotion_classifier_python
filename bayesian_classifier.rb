require 'classifier'

classifier = Classifier::Bayes.new "Angry", "Fearful"

File.open('emotions/angry.txt').each { |grr| classifier.train_angry grr }
File.open('emotions/fearful.txt').each { |eek| classifier.train_fearful eek }

puts classifier.classify "I hate you!"
#=> Angry
puts classifier.classify "Please don't hurt me"
#=> Fearful
