# Generated by Django 5.1.7 on 2025-04-14 20:03

from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='MedicalClaim',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('claim_amount', models.FloatField()),
                ('claim_date', models.DateField()),
                ('procedure_code', models.CharField(max_length=20)),
                ('patient_age', models.IntegerField()),
                ('patient_gender', models.CharField(choices=[('Male', 'Male'), ('Female', 'Female'), ('Other', 'Other')], max_length=10)),
                ('provider_speciality', models.CharField(max_length=100)),
                ('claim_status', models.CharField(max_length=50)),
                ('patient_income', models.FloatField()),
                ('patient_marital_status', models.CharField(choices=[('Single', 'Single'), ('Married', 'Married'), ('Divorced', 'Divorced')], max_length=20)),
                ('patient_employment_status', models.CharField(choices=[('Employed', 'Employed'), ('Unemployed', 'Unemployed'), ('Retired', 'Retired')], max_length=20)),
                ('claim_type', models.CharField(max_length=50)),
                ('submission_method', models.CharField(max_length=50)),
                ('is_fraud', models.BooleanField(blank=True, null=True)),
            ],
        ),
    ]
