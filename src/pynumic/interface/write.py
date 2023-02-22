"""TODO:"""

from typing import Any


def write(obj: object, *args: Any, **kwargs: Any) -> None:
    """TODO:"""
    print(obj, args, kwargs)

# def write_config(obj, filename: str) -> Exception:
#     print(obj, filename)
#     return Exception()
#
#
# def write_weights(obj, filename: str) -> Exception:
#     print(obj, filename)
#     return Exception()

# // WriteConfig writes the configuration and weights to the Filer interface object.
# func (nn *NN) WriteConfig(name ...string) (err error) {
# 	if len(name) > 0 {
# 		switch d := utils.GetFileType(name[0]).(type) {
# 		case error:
# 			err = d
# 		case utils.Filer:
# 			err = d.Encode(nn)
# 		}
# 	} else if nn.config != nil {
# 		err = nn.config.Encode(nn)
# 	} else {
# 		err = pkg.ErrNoArgs
# 	}
#
# 	if err != nil {
# 		err = fmt.Errorf("perceptron.NN.WriteConfig: %w", err)
# 		log.Print(err)
# 	}
# 	return
# }
#
# // WriteWeights writes weights to the Filer interface object.
# func (nn *NN) WriteWeights(name string) (err error) {
# 	switch d := utils.GetFileType(name).(type) {
# 	case error:
# 		err = d
# 	case utils.Filer:
# 		err = d.Encode(nn.Weights)
# 	}
#
# 	if err != nil {
# 		err = fmt.Errorf("perceptron.NN.WriteWeights: %w", err)
# 		log.Print(err)
# 	}
# 	return
# }
