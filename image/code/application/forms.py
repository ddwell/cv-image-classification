from django import forms

class DocumentForm(forms.Form):
    docfile = forms.FileField(
        label='',
        help_text='',
		widget=forms.FileInput(attrs={
			'accept':'image/*', 
			}),
    )